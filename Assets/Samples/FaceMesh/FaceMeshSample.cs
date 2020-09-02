using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;

public sealed class FaceMeshSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string faceModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField, FilePopup("*.tflite")] string faceMeshModelFile = "coco_ssd_mobilenet_quant.tflite";

    [SerializeField] RawImage cameraView = null;
    [SerializeField] RawImage croppedView = null;
    [SerializeField] Material faceMaterial = null;


    WebCamTexture webcamTexture;
    FaceDetect faceDetect;
    FaceMesh faceMesh;
    FaceDetect.Result detectionResult;
    FaceMesh.Result meshResult;
    PrimitiveDraw draw;
    Vector3[] rtCorners = new Vector3[4];
    MeshFilter faceMeshFilter;
    Vector3[] faceKeypoints;


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

        draw = new PrimitiveDraw(Camera.main, gameObject.layer);

        // Create Face Mesh Renderer
        {
            var go = new GameObject("Face");
            go.transform.SetParent(transform);
            var faceRenderer = go.AddComponent<MeshRenderer>();
            faceRenderer.material = faceMaterial;

            faceMeshFilter = go.AddComponent<MeshFilter>();
            faceMeshFilter.sharedMesh = FaceMeshBuilder.CreateMesh();

            faceKeypoints = new Vector3[FaceMesh.KEYPOINT_COUNT];
        }
    }

    void OnDestroy()
    {
        webcamTexture?.Stop();
        faceDetect?.Dispose();
        faceMesh?.Dispose();
        draw?.Dispose();
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

        if (meshResult.score < 0.5f)
        {
            return;
        }

        OnDrawResults();
    }

    void OnDrawResults()
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
        draw.Apply();

        // Draw face
        draw.color = Color.green;
        float zScale = (max.x - min.x) / 2;
        for (int i = 0; i < meshResult.keypoints.Length; i++)
        {
            Vector3 p = MathTF.Leap(min, max, meshResult.keypoints[i]);
            p.z = meshResult.keypoints[i].z * zScale;
            faceKeypoints[i] = p;
            draw.Point(p, 0.05f);
        }
        draw.Apply();

        // Update Mesh
        FaceMeshBuilder.UpdateMesh(faceMeshFilter.sharedMesh, faceKeypoints);
    }
}
