using System.Collections.Generic;
using TensorFlowLite;
using TextureSource;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// MediaPipe Hand Tracking Example
/// https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
/// </summary>
[RequireComponent(typeof(VirtualTextureSource))]
public class HandTrackingSample : MonoBehaviour
{
    [Header("Model Settings")]
    [SerializeField, FilePopup("*.tflite")]
    private string palmModelFile = "coco_ssd_mobilenet_quant.tflite";

    [SerializeField, FilePopup("*.tflite")]
    private string landmarkModelFile = "coco_ssd_mobilenet_quant.tflite";

    [SerializeField]
    private bool useLandmarkToDetection = true;

    [Header("UI")]
    [SerializeField]
    private RawImage inputView = null;
    [SerializeField]
    private RawImage croppedView = null;

    private PalmDetect palmDetect;
    private HandLandmarkDetect landmarkDetect;

    // just cache for GetWorldCorners
    private readonly Vector3[] rtCorners = new Vector3[4];
    private readonly Vector3[] worldJoints = new Vector3[HandLandmarkDetect.JOINT_COUNT];
    private PrimitiveDraw draw;
    private List<PalmDetect.Result> palmResults;
    private HandLandmarkDetect.Result landmarkResult;

    private Material previewMaterial;

    private void Start()
    {
        palmDetect = new PalmDetect(palmModelFile);
        landmarkDetect = new HandLandmarkDetect(landmarkModelFile);
        Debug.Log($"landmark dimension: {landmarkDetect.Dim}");

        draw = new PrimitiveDraw();

        previewMaterial = new Material(Shader.Find("Hidden/TFLite/InputMatrixPreview"));
        inputView.material = previewMaterial;

        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.AddListener(OnTextureUpdate);
        }
    }

    private void OnDestroy()
    {
        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.RemoveListener(OnTextureUpdate);
        }
        palmDetect?.Dispose();
        landmarkDetect?.Dispose();
    }

    private void Update()
    {
        DrawPalms(palmResults, Color.green);

        if (landmarkResult != null && landmarkResult.score > 0.2f)
        {
            DrawJoints(landmarkResult.keypoints);
        }
    }

    private void OnDrawGizmos()
    {
        if (palmResults == null || palmResults.Count == 0)
        {
            return;
        }
        float3 min = rtCorners[0];
        float3 max = rtCorners[2];

        Color startColor = Color.green;
        Color endColor = Color.red;

        foreach (var palm in palmResults)
        {
            int landmarkCount = palm.keypoints.Length;
            for (int i = 0; i < landmarkCount; i++)
            {
                Vector2 kp = palm.keypoints[i];
                Gizmos.color = Color.Lerp(startColor, endColor, i / (float)landmarkCount);
                Gizmos.DrawSphere(math.lerp(min, max, new float3(kp.x, 1 - kp.y, 0)), 0.05f);
            }
        }
    }

    private void OnTextureUpdate(Texture texture)
    {
        bool needPalmDetect = palmResults == null || palmResults.Count == 0 || !useLandmarkToDetection;
        if (needPalmDetect)
        {
            palmDetect.Run(texture);

            inputView.texture = texture;
            previewMaterial.SetMatrix("_TransformMatrix", palmDetect.InputTransformMatrix);
            inputView.rectTransform.GetWorldCorners(rtCorners);

            palmResults = palmDetect.GetResults(0.7f, 0.3f);

            if (palmResults.Count <= 0)
            {
                return;
            };
        }

        // Detect only first palm
        landmarkDetect.Palm = palmResults[0];
        landmarkDetect.Run(texture);
        croppedView.texture = landmarkDetect.InputTexture;
        landmarkResult = landmarkDetect.GetResult();

        if (landmarkResult.score < 0.5f)
        {
            palmResults.Clear();
            return;
        }

        if (useLandmarkToDetection)
        {
            palmResults.Clear();
            palmResults.Add(landmarkResult.ToDetection());
        }
    }

    private void DrawPalms(List<PalmDetect.Result> palms, Color color)
    {
        if (palms == null || palms.Count == 0)
        {
            return;
        }
        float3 min = rtCorners[0];
        float3 max = rtCorners[2];

        draw.color = color;
        foreach (var palm in palms)
        {
            draw.Rect(MathTF.Lerp((Vector3)min, (Vector3)max, palm.rect.FlipY()), 0.02f, min.z);

            foreach (Vector2 kp in palm.keypoints)
            {
                draw.Point(math.lerp(min, max, new float3(kp.x, 1 - kp.y, 0)), 0.05f);
            }
        }
        draw.Apply();
    }

    private void DrawJoints(Vector3[] joints)
    {
        draw.color = Color.blue;

        // Get World Corners
        float3 min = rtCorners[0];
        float3 max = rtCorners[2];

        // Get joint locations in the world space
        float zScale = max.x - min.x;
        for (int i = 0; i < HandLandmarkDetect.JOINT_COUNT; i++)
        {
            float3 p0 = joints[i];
            float3 p1 = math.lerp(min, max, p0);
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
