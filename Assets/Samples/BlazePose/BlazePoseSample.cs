using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;

/// <summary>
/// BlazePose form MediaPipe
/// https://github.com/google/mediapipe
/// https://viz.mediapipe.dev/demo/pose_tracking
/// </summary>
public class BlazePoseSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string poseDetectionModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField, FilePopup("*.tflite")] string poseLandmarkModelFile = "coco_ssd_mobilenet_quant.tflite";

    [SerializeField] RawImage cameraView = null;
    [SerializeField] Image framePrefab = null;
    [SerializeField] RawImage debugView = null;
    [SerializeField] Image croppedFrame = null;
    [SerializeField] Mesh jointMesh = null;
    [SerializeField] Material jointMaterial = null;
    [SerializeField] bool useLandmarkFilter = true;

    WebCamTexture webcamTexture;
    PoseDetect poseDetect;
    PoseLandmarkDetect poseLandmark;

    Image frame;
    Vector3[] rtCorners = new Vector3[4]; // just cache for GetWorldCorners
    Matrix4x4[] jointMatrices = new Matrix4x4[PoseLandmarkDetect.JOINT_COUNT];
    PoseLandmarkDetect.Result landmarkResult;
    Vector3[] worldJoints = new Vector3[PoseLandmarkDetect.JOINT_COUNT];
    PrimitiveDraw draw;

    void Start()
    {
        // Init model
        string detectionPath = Path.Combine(Application.streamingAssetsPath, poseDetectionModelFile);
        poseDetect = new PoseDetect(detectionPath);
        string landmarkPath = Path.Combine(Application.streamingAssetsPath, poseLandmarkModelFile);
        poseLandmark = new PoseLandmarkDetect(landmarkPath);

        // Init camera 
        string cameraName = WebCamUtil.FindName(new WebCamUtil.PreferSpec()
        {
            isFrontFacing = false,
            kind = WebCamKind.WideAngle,
        });
        webcamTexture = new WebCamTexture(cameraName, 1280, 720, 30);
        cameraView.texture = webcamTexture;
        webcamTexture.Play();
        Debug.Log($"Starting camera: {cameraName}");

        // Init frame
        frame = Instantiate(framePrefab, Vector3.zero, Quaternion.identity, cameraView.transform);

        draw = new PrimitiveDraw()
        {
            color = Color.blue,
        };
    }

    void OnDestroy()
    {
        webcamTexture?.Stop();
        poseDetect?.Dispose();
        poseLandmark?.Dispose();
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
        poseDetect.Invoke(webcamTexture);
        cameraView.material = poseDetect.transformMat;

        var pose = poseDetect.GetResults(0.7f, 0.3f);
        UpdateFrame(ref pose);

        if (pose.score < 0)
        {
            return;
        }

        poseLandmark.Invoke(webcamTexture, pose);
        debugView.texture = poseLandmark.inputTex;

        landmarkResult = poseLandmark.GetResult(useLandmarkFilter);
        {
            // Apply webcam rotation to draw landmarks correctly
            Matrix4x4 mtx = WebCamUtil.GetMatrix(-webcamTexture.videoRotationAngle, false, webcamTexture.videoVerticallyMirrored);
            for (int i = 0; i < landmarkResult.joints.Length; i++)
            {
                landmarkResult.joints[i] = mtx.MultiplyPoint3x4(landmarkResult.joints[i]);
            }
        }

        RectTransformationCalculator.ApplyToRectTransform(poseLandmark.CropMatrix, croppedFrame.rectTransform);
    }

    void UpdateFrame(ref PoseDetect.Result pose)
    {
        if (pose.score < 0)
        {
            frame.gameObject.SetActive(false);
            return;
        }
        frame.gameObject.SetActive(true);

        var size = ((RectTransform)cameraView.transform).rect.size;
        var rt = frame.transform as RectTransform;
        var p = pose.rect.position;
        p.y = 1.0f - p.y; // invert Y
        rt.anchoredPosition = p * size - size * 0.5f;
        rt.sizeDelta = pose.rect.size * size;

        // Draw keypoints
        var kpOffset = -rt.anchoredPosition + new Vector2(-rt.sizeDelta.x, rt.sizeDelta.y) * 0.5f;
        for (int i = 0; i < 4; i++)
        {
            var child = (RectTransform)rt.GetChild(i);
            Vector2 kp = pose.keypoints[i];
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
        var rt = cameraView.transform as RectTransform;
        rt.GetWorldCorners(rtCorners);
        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];
        float zScale = max.x - min.x;
        for (int i = 0; i < joints.Length; i++)
        {
            var p = joints[i];
            p = MathTF.Leap3(min, max, p);
            p.z += (joints[i].z - 0.5f) * zScale;

            worldJoints[i] = p;
        }

        // Draw
        for (int i = 0; i < worldJoints.Length; i++)
        {
            draw.Cube(worldJoints[i], 0.1f);
        }
        var connections = PoseLandmarkDetect.CONNECTIONS;
        for (int i = 0; i < connections.Length; i += 2)
        {
            draw.Line(
                worldJoints[connections[i]],
                worldJoints[connections[i + 1]],
                0.05f);
        }

    }
}