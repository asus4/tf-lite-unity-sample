using System.Threading;
using Cysharp.Threading.Tasks;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// BlazePose form MediaPipe
/// https://github.com/google/mediapipe
/// https://viz.mediapipe.dev/demo/pose_tracking
/// </summary>
[RequireComponent(typeof(WebCamInput))]
public sealed class BlazePoseSample : MonoBehaviour
{

    [SerializeField, FilePopup("*.tflite")]
    private string poseDetectionModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField, FilePopup("*.tflite")]
    private string poseLandmarkModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField]
    private RawImage cameraView = null;
    [SerializeField]
    private RawImage debugView = null;
    [SerializeField]
    private Canvas canvas = null;
    [SerializeField]
    private bool runBackground;
    [SerializeField, Range(0f, 1f)]
    private float visibilityThreshold = 0.5f;
    [SerializeField]
    private PoseLandmarkDetect.Options landmarkOptions = new PoseLandmarkDetect.Options();

    private readonly Vector3[] rtCorners = new Vector3[4]; // just cache for GetWorldCorners

    private PoseDetect poseDetect;
    private PoseLandmarkDetect poseLandmark;

    private Vector4[] viewportLandmarks;
    private PrimitiveDraw draw;
    private PoseDetect.Result poseResult;
    private PoseLandmarkDetect.Result landmarkResult;
    private UniTask<bool> task;
    private CancellationToken cancellationToken;

    private bool NeedsDetectionUpdate => poseResult == null || poseResult.score < 0.5f;

    private void Start()
    {
        // Init model
        poseDetect = new PoseDetect(poseDetectionModelFile);
        poseLandmark = new PoseLandmarkDetect(poseLandmarkModelFile, landmarkOptions);
     
        draw = new PrimitiveDraw(Camera.main, gameObject.layer);
        viewportLandmarks = new Vector4[PoseLandmarkDetect.LandmarkCount];

        cancellationToken = this.GetCancellationTokenOnDestroy();

        GetComponent<WebCamInput>().OnTextureUpdate.AddListener(OnTextureUpdate);
    }

    private void OnDestroy()
    {
        GetComponent<WebCamInput>().OnTextureUpdate.RemoveListener(OnTextureUpdate);

        poseDetect?.Dispose();
        poseLandmark?.Dispose();
        draw?.Dispose();
    }

    private void OnTextureUpdate(Texture texture)
    {
        if (runBackground)
        {
            if (task.Status.IsCompleted())
            {
                task = InvokeAsync(texture);
            }
        }
        else
        {
            Invoke(texture);
        }
    }

    private void Update()
    {
        if (poseResult != null && poseResult.score > 0f)
        {
            DrawFrame(poseResult);
        }

        if (landmarkResult != null && landmarkResult.score > 0.2f)
        {
            DrawCropMatrix(poseLandmark.CropMatrix);
            DrawViewportLandmarks(landmarkResult.viewportLandmarks);
            if (landmarkOptions.useWorldLandmarks)
            {
                DrawWorldLandmarks(landmarkResult.worldLandmarks);
            }
        }
    }

    void DrawFrame(PoseDetect.Result pose)
    {
        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];

        draw.color = Color.green;
        draw.Rect(MathTF.Lerp(min, max, pose.rect, true), 0.02f, min.z);

        foreach (var kp in pose.keypoints)
        {
            draw.Point(MathTF.Lerp(min, max, (Vector3)kp, true), 0.05f);
        }
        draw.Apply();
    }

    private void DrawCropMatrix(in Matrix4x4 matrix)
    {
        draw.color = Color.red;

        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];

        Matrix4x4 mtx = matrix.inverse;
        Vector3 a = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(0, 0, 0)));
        Vector3 b = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(1, 0, 0)));
        Vector3 c = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(1, 1, 0)));
        Vector3 d = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(0, 1, 0)));

        draw.Quad(a, b, c, d, 0.02f);
        draw.Apply();
    }

    private void DrawViewportLandmarks(Vector4[] landmarks)
    {
        draw.color = Color.blue;

        float zScale = 1;
        float zOffset = canvas.planeDistance;
        float aspect = (float)Screen.width / Screen.height;
        Vector3 scale, offset;
        if (aspect > 1)
        {
            scale = new Vector3(1f / aspect, 1f, zScale);
            offset = new Vector3((1 - 1f / aspect) / 2, 0, zOffset);
        }
        else
        {
            scale = new Vector3(1f, aspect, zScale);
            offset = new Vector3(0, (1 - aspect) / 2, zOffset);
        }

        // Update world joints
        var camera = canvas.worldCamera;
        for (int i = 0; i < landmarks.Length; i++)
        {
            Vector3 p = Vector3.Scale(landmarks[i], scale) + offset;
            p = camera.ViewportToWorldPoint(p);

            // w is visibility
            viewportLandmarks[i] = new Vector4(p.x, p.y, p.z, landmarks[i].w);
        }

        // Draw
        for (int i = 0; i < viewportLandmarks.Length; i++)
        {
            Vector4 p = viewportLandmarks[i];
            if (p.w > visibilityThreshold)
            {
                draw.Cube(p, 0.2f);
            }
        }
        var connections = PoseLandmarkDetect.Connections;
        for (int i = 0; i < connections.Length; i += 2)
        {
            var a = viewportLandmarks[connections[i]];
            var b = viewportLandmarks[connections[i + 1]];
            if (a.w > visibilityThreshold || b.w > visibilityThreshold)
            {
                draw.Line3D(a, b, 0.05f);
            }
        }
        draw.Apply();
    }

    private void DrawWorldLandmarks(Vector4[] landmarks)
    {
        draw.color = Color.cyan;

        for (int i = 0; i < landmarks.Length; i++)
        {
            Vector4 p = landmarks[i];
            if (p.w > visibilityThreshold)
            {
                draw.Cube(p, 0.02f);
            }
        }
        var connections = PoseLandmarkDetect.Connections;
        for (int i = 0; i < connections.Length; i += 2)
        {
            var a = landmarks[connections[i]];
            var b = landmarks[connections[i + 1]];
            if (a.w > visibilityThreshold || b.w > visibilityThreshold)
            {
                draw.Line3D(a, b, 0.005f);
            }
        }

        draw.Apply();
    }

    private void Invoke(Texture texture)
    {
        if (NeedsDetectionUpdate)
        {
            poseDetect.Invoke(texture);
            cameraView.material = poseDetect.transformMat;
            cameraView.rectTransform.GetWorldCorners(rtCorners);
            poseResult = poseDetect.GetResults(0.7f, 0.3f);
        }
        if (poseResult.score < 0)
        {
            poseResult = null;
            landmarkResult = null;
            return;
        }
        poseLandmark.Invoke(texture, poseResult);
        debugView.texture = poseLandmark.inputTex;

        landmarkResult = poseLandmark.GetResult();

        if (landmarkResult.score < 0.3f)
        {
            poseResult.score = landmarkResult.score;
        }
        else
        {
            poseResult = PoseLandmarkDetect.LandmarkToDetection(landmarkResult);
        }
    }

    private async UniTask<bool> InvokeAsync(Texture texture)
    {
        if (NeedsDetectionUpdate)
        {
            // Note: `await` changes PlayerLoopTiming from Update to FixedUpdate.
            poseResult = await poseDetect.InvokeAsync(texture, cancellationToken, PlayerLoopTiming.FixedUpdate);
        }
        if (poseResult.score < 0)
        {
            poseResult = null;
            landmarkResult = null;
            return false;
        }

        landmarkResult = await poseLandmark.InvokeAsync(texture, poseResult, cancellationToken, PlayerLoopTiming.Update);

        // Back to the update timing from now on 
        if (cameraView != null)
        {
            cameraView.material = poseDetect.transformMat;
            cameraView.rectTransform.GetWorldCorners(rtCorners);
        }
        if (debugView != null)
        {
            debugView.texture = poseLandmark.inputTex;
        }

        // Generate poseResult from landmarkResult
        if (landmarkResult.score < 0.3f)
        {
            poseResult.score = landmarkResult.score;
        }
        else
        {
            poseResult = PoseLandmarkDetect.LandmarkToDetection(landmarkResult);
        }

        return true;
    }
}
