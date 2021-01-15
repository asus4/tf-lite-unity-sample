using System.Collections.Generic;
using System.Threading;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;
using Cysharp.Threading.Tasks;

public class HandTrackingSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string palmModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField, FilePopup("*.tflite")] string landmarkModelFile = "coco_ssd_mobilenet_quant.tflite";

    [SerializeField] RawImage cameraView = null;
    [SerializeField] RawImage debugPalmView = null;
    [SerializeField] bool runBackground;

    WebCamTexture webcamTexture;
    PalmDetect palmDetect;
    HandLandmarkDetect landmarkDetect;

    // just cache for GetWorldCorners
    Vector3[] rtCorners = new Vector3[4];
    Vector3[] worldJoints = new Vector3[HandLandmarkDetect.JOINT_COUNT];
    PrimitiveDraw draw;
    List<PalmDetect.Result> palmResults;
    HandLandmarkDetect.Result landmarkResult;
    UniTask<bool> task;
    CancellationToken cancellationToken;



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

        draw = new PrimitiveDraw();
    }

    void OnDestroy()
    {
        webcamTexture?.Stop();
        palmDetect?.Dispose();
        landmarkDetect?.Dispose();
    }

    void Update()
    {
        if (runBackground)
        {
            if (task.Status.IsCompleted())
            {
                task = InvokeAsync();
            }
        }
        else
        {
            Invoke();
        }

        if (palmResults == null || palmResults.Count <= 0) return;
        DrawFrames(palmResults);

        if (landmarkResult == null || landmarkResult.score < 0.2f) return;
        DrawCropMatrix(landmarkDetect.CropMatrix);
        DrawJoints(landmarkResult.joints);
    }

    void Invoke()
    {
        palmDetect.Invoke(webcamTexture);
        cameraView.material = palmDetect.transformMat;
        cameraView.rectTransform.GetWorldCorners(rtCorners);

        palmResults = palmDetect.GetResults(0.7f, 0.3f);


        if (palmResults.Count <= 0) return;

        // Detect only first palm
        landmarkDetect.Invoke(webcamTexture, palmResults[0]);
        debugPalmView.texture = landmarkDetect.inputTex;

        landmarkResult = landmarkDetect.GetResult();
    }

    async UniTask<bool> InvokeAsync()
    {
        palmResults = await palmDetect.InvokeAsync(webcamTexture, cancellationToken);
        cameraView.material = palmDetect.transformMat;
        cameraView.rectTransform.GetWorldCorners(rtCorners);

        if (palmResults.Count <= 0) return false;

        landmarkResult = await landmarkDetect.InvokeAsync(webcamTexture, palmResults[0], cancellationToken);
        debugPalmView.texture = landmarkDetect.inputTex;

        return true;
    }

    void DrawFrames(List<PalmDetect.Result> palms)
    {
        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];

        draw.color = Color.green;
        foreach (var palm in palms)
        {
            draw.Rect(MathTF.Lerp(min, max, palm.rect, true), 0.02f, min.z);

            foreach (var kp in palm.keypoints)
            {
                draw.Point(MathTF.Lerp(min, max, (Vector3)kp, true), 0.05f);
            }
        }
        draw.Apply();
    }

    void DrawCropMatrix(in Matrix4x4 matrix)
    {
        draw.color = Color.red;

        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];

        var mtx = WebCamUtil.GetMatrix(-webcamTexture.videoRotationAngle, false, webcamTexture.videoVerticallyMirrored)
            * matrix.inverse;
        Vector3 a = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(0, 0, 0)));
        Vector3 b = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(1, 0, 0)));
        Vector3 c = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(1, 1, 0)));
        Vector3 d = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(0, 1, 0)));

        draw.Quad(a, b, c, d, 0.02f);
        draw.Apply();
    }

    void DrawJoints(Vector3[] joints)
    {
        draw.color = Color.blue;

        // Get World Corners
        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];

        // Need to apply camera rotation and mirror on mobile
        Matrix4x4 mtx = WebCamUtil.GetMatrix(-webcamTexture.videoRotationAngle, false, webcamTexture.videoVerticallyMirrored);

        // Get joint locations in the world space
        float zScale = max.x - min.x;
        for (int i = 0; i < HandLandmarkDetect.JOINT_COUNT; i++)
        {
            Vector3 p0 = mtx.MultiplyPoint3x4(joints[i]);
            Vector3 p1 = MathTF.Lerp(min, max, p0);
            p1.z += (p0.z - 0.5f) * zScale;
            worldJoints[i] = p1;
        }

        // Cube
        for (int i = 0; i < HandLandmarkDetect.JOINT_COUNT; i++)
        {
            draw.Cube(worldJoints[i], 0.1f);
        }

        // Connection Lines
        var connections = HandLandmarkDetect.CONNECTIONS;
        for (int i = 0; i < connections.Length; i += 2)
        {
            draw.Line3D(
                worldJoints[connections[i]],
                worldJoints[connections[i + 1]],
                0.05f);
        }

        draw.Apply();
    }

}
