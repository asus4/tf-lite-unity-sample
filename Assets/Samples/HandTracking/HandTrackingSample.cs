using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;

public class HandTrackingSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string palmModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField, FilePopup("*.tflite")] string landmarkModelFile = "coco_ssd_mobilenet_quant.tflite";

    [SerializeField] RawImage cameraView = null;
    [SerializeField] Image framePrefab = null;
    [SerializeField] RawImage debugPalmView = null;
    [SerializeField] Image cropedFrame = null;
    [SerializeField] Mesh jointMesh = null;
    [SerializeField] Material jointMaterial = null;
    WebCamTexture webcamTexture;
    PalmDetect palmDetect;
    HandLandmarkDetect landmarkDetect;

    Image[] frames;
    // just cache for GetWorldCorners
    Vector3[] rtCorners = new Vector3[4];
    HandLandmarkDetect.Result landmarkResult;
    Vector3[] worldJoints = new Vector3[HandLandmarkDetect.JOINT_COUNT];
    PrimitiveDraw draw;

    void Start()
    {
        string palmPath = Path.Combine(Application.streamingAssetsPath, palmModelFile);
        palmDetect = new PalmDetect(palmPath);

        string landmarkPath = Path.Combine(Application.streamingAssetsPath, landmarkModelFile);
        landmarkDetect = new HandLandmarkDetect(landmarkPath);
        Debug.Log($"landmark dimension: {landmarkDetect.Dim}");

        string cameraName = WebCamUtil.FindName(new WebCamUtil.PreferSpec()
        {
            isFrontFacing = false,
            kind = WebCamKind.WideAngle,
        });
        webcamTexture = new WebCamTexture(cameraName, 1280, 720, 30);
        cameraView.texture = webcamTexture;
        webcamTexture.Play();
        Debug.Log($"Starting camera: {cameraName}");

        // Init frames
        frames = new Image[PalmDetect.MAX_PALM_NUM];
        var parent = cameraView.transform;
        for (int i = 0; i < frames.Length; i++)
        {
            frames[i] = Instantiate(framePrefab, Vector3.zero, Quaternion.identity, parent);
            frames[i].transform.localPosition = Vector3.zero;
        }

        draw = new PrimitiveDraw()
        {
            color = Color.blue,
        };
    }
    void OnDestroy()
    {
        webcamTexture?.Stop();
        palmDetect?.Dispose();
        landmarkDetect?.Dispose();
    }

    void OnEnable()
    {
        Camera.onPostRender += DrawJoints;
    }
    void OnDisable()
    {
        Camera.onPostRender -= DrawJoints;
    }

    void Update()
    {


        palmDetect.Invoke(webcamTexture);
        cameraView.material = palmDetect.transformMat;

        var palms = palmDetect.GetResults(0.7f, 0.3f);
        UpdateFrame(palms);

        if (palms.Count <= 0)
        {
            if (landmarkResult != null)
            {
                landmarkResult.score = 0;
            }
            return;
        }

        // Detect only first palm
        landmarkDetect.Invoke(webcamTexture, palms[0]);
        debugPalmView.texture = landmarkDetect.inputTex;

        landmarkResult = landmarkDetect.GetResult();
        {
            // Apply webcam rotation to draw landmarks correctly
            Matrix4x4 mtx = WebCamUtil.GetMatrix(-webcamTexture.videoRotationAngle, false, webcamTexture.videoVerticallyMirrored);
            for (int i = 0; i < landmarkResult.joints.Length; i++)
            {
                landmarkResult.joints[i] = mtx.MultiplyPoint3x4(landmarkResult.joints[i]);
            }
        }

        RectTransformationCalculator.ApplyToRectTransform(landmarkDetect.CropMatrix, cropedFrame.rectTransform);
    }

    void UpdateFrame(List<PalmDetect.Result> palms)
    {
        var size = ((RectTransform)cameraView.transform).rect.size;
        for (int i = 0; i < palms.Count; i++)
        {
            frames[i].gameObject.SetActive(true);
            SetFrame(frames[i], palms[i], size);
        }
        for (int i = palms.Count; i < frames.Length; i++)
        {
            frames[i].gameObject.SetActive(false);
        }
    }


    void SetFrame(Graphic frame, PalmDetect.Result palm, Vector2 size)
    {
        var rt = frame.transform as RectTransform;
        var p = palm.rect.position;
        p.y = 1.0f - p.y; // invert Y
        rt.anchoredPosition = p * size - size * 0.5f;
        rt.sizeDelta = palm.rect.size * size;

        var kpOffset = -rt.anchoredPosition + new Vector2(-rt.sizeDelta.x, rt.sizeDelta.y) * 0.5f;
        for (int i = 0; i < 7; i++)
        {
            var child = (RectTransform)rt.GetChild(i);
            var kp = palm.keypoints[i];
            kp.y = 1.0f - kp.y; // invert Y
            child.anchoredPosition = (kp * size - size * 0.5f) + kpOffset;
        }
    }

    void DrawJoints(Camera camera)
    {
        if (landmarkResult == null || landmarkResult.score < 0.2f)
        {
            return;
        }

        // Get world position of the joints
        var joints = landmarkResult.joints;
        cameraView.rectTransform.GetWorldCorners(rtCorners);
        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];
        float zScale = max.x - min.x;
        for (int i = 0; i < HandLandmarkDetect.JOINT_COUNT; i++)
        {
            var p = joints[i];
            p = MathTF.Leap3(min, max, p);
            p.z += (joints[i].z - 0.5f) * zScale;
            worldJoints[i] = p;
        }


        for (int i = 0; i < HandLandmarkDetect.JOINT_COUNT; i++)
        {
            draw.Cube(worldJoints[i], 0.1f);
        }

        var connections = HandLandmarkDetect.CONNECTIONS;
        for (int i = 0; i < connections.Length; i += 2)
        {
            draw.Line(
                worldJoints[connections[i]],
                worldJoints[connections[i + 1]],
                0.05f);
        }
    }

}
