using System.Collections.Generic;
using TensorFlowLite;
using TextureSource;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// BlazeFace from MediaPile
/// https://github.com/google/mediapipe
/// https://viz.mediapipe.dev/demo/face_detection
/// </summary>
[RequireComponent(typeof(VirtualTextureSource))]
public class FaceDetectionSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")]
    private string faceModelFile = "coco_ssd_mobilenet_quant.tflite";

    [SerializeField]
    private RawImage inputPreview = null;

    private FaceDetect faceDetect;
    private List<FaceDetect.Result> results;
    private PrimitiveDraw draw;
    private readonly Vector3[] rtCorners = new Vector3[4];
    private Material previewMaterial;

    private void Start()
    {
        faceDetect = new FaceDetect(faceModelFile);
        draw = new PrimitiveDraw(Camera.main, gameObject.layer);

        previewMaterial = new Material(Shader.Find("Hidden/TFLite/InputMatrixPreview"));
        inputPreview.material = previewMaterial;

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
        faceDetect?.Dispose();
        draw?.Dispose();
        Destroy(previewMaterial);
    }

    private void Update()
    {
        DrawResults(results);
    }

    private void OnTextureUpdate(Texture texture)
    {
        faceDetect.Run(texture);

        inputPreview.texture = texture;
        previewMaterial.SetMatrix("_TransformMatrix", faceDetect.InputTransformMatrix);

        inputPreview.rectTransform.GetWorldCorners(rtCorners);
        results = faceDetect.GetResults();
    }

    private void DrawResults(List<FaceDetect.Result> results)
    {
        if (results == null || results.Count == 0)
        {
            return;
        }

        float3 min = rtCorners[0];
        float3 max = rtCorners[2];

        draw.color = Color.blue;

        foreach (var result in results)
        {
            Rect rect = MathTF.Lerp((Vector3)min, (Vector3)max, result.rect.FlipY());
            draw.Rect(rect, 0.05f, -0.1f);
            foreach (Vector2 p in result.keypoints)
            {
                draw.Point(math.lerp(min, max, new float3(p.x, 1f - p.y, 0)), -0.1f);
            }
        }
        draw.Apply();
    }
}
