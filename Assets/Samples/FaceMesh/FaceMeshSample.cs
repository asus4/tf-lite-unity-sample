using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;

public class FaceMeshSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string faceModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField, FilePopup("*.tflite")] string faceMeshModelFile = "coco_ssd_mobilenet_quant.tflite";

    [SerializeField] RawImage cameraView = null;
    [SerializeField] RawImage croppedView = null;


    WebCamTexture webcamTexture;
    FaceDetect faceDetect;
    FaceMesh faceMesh;
    FaceDetect.Result detectionResult;
    FaceMesh.Result meshResult;
    PrimitiveDraw draw;
    Vector3[] rtCorners = new Vector3[4];

    void Start()
    {
        string detectionPath = Path.Combine(Application.streamingAssetsPath, faceModelFile);
        faceDetect = new FaceDetect(detectionPath);

        string faceMeshPath = Path.Combine(Application.streamingAssetsPath, faceMeshModelFile);
        faceMesh = new FaceMesh(faceMeshPath);

        string cameraName = WebCamUtil.FindName(new WebCamUtil.PreferSpec()
        {
            isFrontFacing = false,
            kind = WebCamKind.WideAngle,
        });
        webcamTexture = new WebCamTexture(cameraName, 1280, 720, 30);
        cameraView.texture = webcamTexture;
        webcamTexture.Play();
        Debug.Log($"Starting camera: {cameraName}");

        draw = new PrimitiveDraw()
        {
            color = Color.blue,
        };
    }

    void OnDestroy()
    {
        webcamTexture?.Stop();
        faceDetect?.Dispose();
        faceMesh?.Dispose();
        draw?.Dispose();
    }

    void OnEnable()
    {
        Camera.onPostRender += OnDrawResults;
    }
    void OnDisable()
    {
        Camera.onPostRender -= OnDrawResults;
    }


    void Update()
    {
        faceDetect.Invoke(webcamTexture);
        cameraView.material = faceDetect.transformMat;
        detectionResult = faceDetect.GetResults().FirstOrDefault();

        if (detectionResult == null)
        {
            return;
        }

        faceMesh.Invoke(webcamTexture, detectionResult);
        croppedView.texture = faceMesh.inputTex;
        meshResult = faceMesh.GetResult();
    }

    void OnDrawResults(Camera camera)
    {
        if (detectionResult == null)
        {
            return;
        }

        cameraView.rectTransform.GetWorldCorners(rtCorners);
        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];

        // Draw Face Detection
        {
            draw.color = Color.blue;
            Rect rect = MathTF.Leap(min, max, detectionResult.rect, true);
            draw.Rect(rect, 0.05f);
            foreach (Vector2 p in detectionResult.keypoints)
            {
                draw.Point(MathTF.Leap(min, max, new Vector3(p.x, 1f - p.y, 0)), 0.1f);
            }
        }

        {
            float zScale = max.x - min.x;
            draw.color = Color.green;
            foreach (Vector3 p in meshResult.keypoints)
            {
                Vector3 p1 = MathTF.Leap(min, max, p);
                p1.z += (p.z - 0.5f) * zScale;
                draw.Point(p1, 0.05f);
            }
        }
    }


}
