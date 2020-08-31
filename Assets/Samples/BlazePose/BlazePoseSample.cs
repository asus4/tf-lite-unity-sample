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
public sealed class BlazePoseSample : MonoBehaviour
{
    public enum Mode
    {
        UpperBody,
        FullBody,
    }

    [SerializeField, FilePopup("*.tflite")] string poseDetectionModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField, FilePopup("*.tflite")] string poseLandmarkModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField] Mode mode = Mode.UpperBody;
    [SerializeField] RawImage cameraView = null;
    [SerializeField] Image framePrefab = null;
    [SerializeField] RawImage debugView = null;
    [SerializeField] Image croppedFrame = null;
    [SerializeField] bool useLandmarkFilter = true;
    [SerializeField] Vector2 poseShift;
    [SerializeField] Vector2 poseScale;
    [SerializeField, Range(2f, 30f)] float filterVelocityScale = 10;

    WebCamTexture webcamTexture;
    PoseDetect poseDetect;
    PoseLandmarkDetect poseLandmark;

    Image frame;
    Vector3[] rtCorners = new Vector3[4]; // just cache for GetWorldCorners
    PoseLandmarkDetect.Result landmarkResult;
    Vector3[] worldJoints;
    PrimitiveDraw draw;

    void Start()
    {
        // Init model
        string detectionPath = Path.Combine(Application.streamingAssetsPath, poseDetectionModelFile);
        string landmarkPath = Path.Combine(Application.streamingAssetsPath, poseLandmarkModelFile);
        switch (mode)
        {
            case Mode.UpperBody:
                poseDetect = new PoseDetectUpperBody(detectionPath);
                poseLandmark = new PoseLandmarkDetectUpperBody(landmarkPath);
                break;
            case Mode.FullBody:
                poseDetect = new PoseDetectFullBody(detectionPath);
                poseLandmark = new PoseLandmarkDetectFullBody(landmarkPath);
                break;
            default:
                throw new System.NotSupportedException($"Mode: {mode} is not supported");
        }

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
        worldJoints = new Vector3[poseLandmark.JointCount];

        poseShift = poseLandmark.PoseShift;
        poseScale = poseLandmark.PoseScale;
    }

    void OnDestroy()
    {
        webcamTexture?.Stop();
        poseDetect?.Dispose();
        poseLandmark?.Dispose();
        draw?.Dispose();
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

        poseLandmark.PoseShift = poseShift;
        poseLandmark.PoseScale = poseScale;

        poseLandmark.Invoke(webcamTexture, pose);
        debugView.texture = poseLandmark.inputTex;

        if (useLandmarkFilter)
        {
            poseLandmark.FilterVelocityScale = filterVelocityScale;
        }
        landmarkResult = poseLandmark.GetResult(useLandmarkFilter);
        UpdateJoints();
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
        for (int i = 0; i < poseDetect.KeypointsCount; i++)
        {
            var child = (RectTransform)rt.GetChild(i);
            Vector2 kp = pose.keypoints[i];
            kp.y = 1.0f - kp.y; // invert Y
            child.anchoredPosition = (kp * size - size * 0.5f) + kpOffset;
        }
    }

    void UpdateJoints()
    {
        // Apply webcam rotation to draw landmarks correctly
        Matrix4x4 mtx = WebCamUtil.GetMatrix(-webcamTexture.videoRotationAngle, false, webcamTexture.videoVerticallyMirrored);
        var rt = cameraView.transform as RectTransform;
        rt.GetWorldCorners(rtCorners);
        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];

        var joints = landmarkResult.joints;
        for (int i = 0; i < joints.Length; i++)
        {
            var p = mtx.MultiplyPoint3x4(joints[i]);
            p = MathTF.Leap(min, max, p);
            worldJoints[i] = p;
        }
    }

    void DrawJoints(Camera camera)
    {
        if (landmarkResult == null || landmarkResult.score < 0.2f)
        {
            return;
        }

        // Draw
        for (int i = 0; i < worldJoints.Length; i++)
        {
            draw.Cube(worldJoints[i], 0.1f);
        }
        var connections = poseLandmark.Connections;
        for (int i = 0; i < connections.Length; i += 2)
        {
            draw.Line3D(
                worldJoints[connections[i]],
                worldJoints[connections[i + 1]],
                0.05f);
        }
    }
}
