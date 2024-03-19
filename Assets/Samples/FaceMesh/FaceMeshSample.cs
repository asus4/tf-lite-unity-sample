using System.Linq;
using TensorFlowLite;
using TextureSource;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(VirtualTextureSource))]
public sealed class FaceMeshSample : MonoBehaviour
{
    [Header("Model Settings")]
    [SerializeField, FilePopup("*.tflite")]
    private string faceModelFile = "coco_ssd_mobilenet_quant.tflite";

    [SerializeField, FilePopup("*.tflite")]
    private string faceMeshModelFile = "coco_ssd_mobilenet_quant.tflite";

    [SerializeField]
    private bool useLandmarkToDetection = true;

    [Header("UI")]
    [SerializeField]
    private RawImage inputPreview = null;

    [SerializeField]
    private RawImage croppedView = null;

    [SerializeField]
    private Material faceMaterial = null;

    private FaceDetect faceDetect;
    private FaceMesh faceMesh;
    private PrimitiveDraw draw;
    private MeshFilter faceMeshFilter;
    private Vector3[] faceVertices;
    private FaceDetect.Result detectionResult;
    private FaceMesh.Result meshResult;
    private readonly Vector3[] rtCorners = new Vector3[4];
    private Material previewMaterial;


    private void Start()
    {
        faceDetect = new FaceDetect(faceModelFile);
        faceMesh = new FaceMesh(faceMeshModelFile);
        draw = new PrimitiveDraw(Camera.main, gameObject.layer);

        previewMaterial = new Material(Shader.Find("Hidden/TFLite/InputMatrixPreview"));
        inputPreview.material = previewMaterial;

        // Create Face Mesh Renderer
        {
            var go = new GameObject("Face");
            go.transform.SetParent(transform);
            var faceRenderer = go.AddComponent<MeshRenderer>();
            faceRenderer.material = faceMaterial;

            faceMeshFilter = go.AddComponent<MeshFilter>();
            faceMeshFilter.sharedMesh = FaceMeshBuilder.CreateMesh();

            faceVertices = new Vector3[FaceMesh.KEYPOINT_COUNT];
        }

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
        faceMesh?.Dispose();
        draw?.Dispose();
        Destroy(previewMaterial);
    }

    private void Update()
    {
        DrawResults(detectionResult, meshResult);
    }

    private void OnTextureUpdate(Texture texture)
    {
        if (detectionResult == null || !useLandmarkToDetection)
        {
            faceDetect.Run(texture);

            inputPreview.texture = texture;
            previewMaterial.SetMatrix("_TransformMatrix", faceDetect.InputTransformMatrix);

            detectionResult = faceDetect.GetResults().FirstOrDefault();

            if (detectionResult == null)
            {
                return;
            }
        }

        faceMesh.Face = detectionResult;
        faceMesh.Run(texture);
        croppedView.texture = faceMesh.InputTexture;
        meshResult = faceMesh.GetResult();

        if (meshResult.score < 0.5f)
        {
            detectionResult = null;
            return;
        }

        if (useLandmarkToDetection)
        {
            detectionResult = meshResult.ToDetection();
        }
    }

    private void DrawResults(FaceDetect.Result detection, FaceMesh.Result face)
    {
        inputPreview.rectTransform.GetWorldCorners(rtCorners);
        float3 min = rtCorners[0];
        float3 max = rtCorners[2];

        // Draw Face Detection
        if (detection != null)
        {
            draw.color = Color.blue;
            Rect rect = MathTF.Lerp((Vector3)min, (Vector3)max, detection.rect.FlipY());
            draw.Rect(rect, 0.05f);
            foreach (Vector2 p in detection.keypoints)
            {
                draw.Point(math.lerp(min, max, new float3(p.x, 1f - p.y, 0)), 0.1f);
            }
            draw.Apply();
        }

        if (face != null)
        {
            // Draw face
            draw.color = Color.green;
            float zScale = (max.x - min.x) / 2;
            for (int i = 0; i < face.keypoints.Length; i++)
            {
                float3 kp = face.keypoints[i];
                float3 p = math.lerp(min, max, kp);
                // TODO: projection is not correct
                p.z = kp.z * zScale;

                faceVertices[i] = p;
                draw.Point(p, 0.05f);
            }
            draw.Apply();

            // Update Mesh
            FaceMeshBuilder.UpdateMesh(faceMeshFilter.sharedMesh, faceVertices);
        }
    }
}
