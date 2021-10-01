using System.Collections.Generic;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// BlazeFace from MediaPile
/// https://github.com/google/mediapipe
/// https://viz.mediapipe.dev/demo/face_detection
/// </summary>
[RequireComponent(typeof(WebCamInput))]
public class FaceDetectionSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")]
    private string faceModelFile = "coco_ssd_mobilenet_quant.tflite";

    [SerializeField]
    private RawImage cameraView = null;

    private FaceDetect faceDetect;
    private List<FaceDetect.Result> results;
    private PrimitiveDraw draw;
    private readonly Vector3[] rtCorners = new Vector3[4];

    private void Start()
    {
        faceDetect = new FaceDetect(faceModelFile);
        draw = new PrimitiveDraw(Camera.main, gameObject.layer);

        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.AddListener(OnTextureUpdate);
    }

    private void OnDestroy()
    {
        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.RemoveListener(OnTextureUpdate);

        faceDetect?.Dispose();
        draw?.Dispose();
    }

    private void Update()
    {
        DrawResults(results);
    }

    private void OnTextureUpdate(Texture texture)
    {
        faceDetect.Invoke(texture);
        cameraView.material = faceDetect.transformMat;
        cameraView.rectTransform.GetWorldCorners(rtCorners);
        results = faceDetect.GetResults();
    }

    private void DrawResults(List<FaceDetect.Result> results)
    {
        if (results == null || results.Count == 0)
        {
            return;
        }

        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];

        draw.color = Color.blue;

        foreach (var result in results)
        {
            Rect rect = MathTF.Lerp(min, max, result.rect, true);
            draw.Rect(rect, 0.05f);
            foreach (Vector2 p in result.keypoints)
            {
                draw.Point(MathTF.Lerp(min, max, new Vector3(p.x, 1f - p.y, 0)), 0.1f);
            }
        }
        draw.Apply();
    }
}
