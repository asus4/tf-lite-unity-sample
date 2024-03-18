using System.Threading;
using Cysharp.Threading.Tasks;
using TensorFlowLite;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(WebCamInput))]
[System.Obsolete("Use MoveNet instead")]
public class PoseNetSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")]
    private string fileName = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite";

    [SerializeField]
    private RawImage cameraView = null;

    [SerializeField, Range(0f, 1f)]
    private float threshold = 0.5f;

    [SerializeField, Range(0f, 1f)]
    private float lineThickness = 0.5f;

    [SerializeField]
    private bool runBackground;

    private PoseNet poseNet;
    private readonly Vector3[] rtCorners = new Vector3[4];
    private PrimitiveDraw draw;
    private UniTask<bool> task;
    private PoseNet.Result[] results;
    private CancellationToken cancellationToken;

    private void Start()
    {
        poseNet = new PoseNet(fileName);

        draw = new PrimitiveDraw(Camera.main, gameObject.layer)
        {
            color = Color.green,
        };

        cancellationToken = this.GetCancellationTokenOnDestroy();

        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.AddListener(OnTextureUpdate);
    }

    private void OnDestroy()
    {
        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.RemoveListener(OnTextureUpdate);

        poseNet?.Dispose();
        draw?.Dispose();
    }

    private void Update()
    {
        DrawResult(results);
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

    private void DrawResult(PoseNet.Result[] results)
    {
        if (results == null || results.Length == 0)
        {
            return;
        }

        var rect = cameraView.GetComponent<RectTransform>();
        rect.GetWorldCorners(rtCorners);
        float3 min = rtCorners[0];
        float3 max = rtCorners[2];

        var connections = PoseNet.Connections;
        int len = connections.GetLength(0);
        for (int i = 0; i < len; i++)
        {
            var a = results[(int)connections[i, 0]];
            var b = results[(int)connections[i, 1]];
            if (a.confidence >= threshold && b.confidence >= threshold)
            {
                draw.Line3D(
                    math.lerp(min, max, new float3(a.x, 1f - a.y, 0)),
                    math.lerp(min, max, new float3(b.x, 1f - b.y, 0)),
                    lineThickness
                );
            }
        }

        draw.Apply();
    }

    private void Invoke(Texture texture)
    {
        poseNet.Invoke(texture);
        results = poseNet.GetResults();
        cameraView.material = poseNet.transformMat;
    }

    private async UniTask<bool> InvokeAsync(Texture texture)
    {
        results = await poseNet.InvokeAsync(texture, cancellationToken);
        cameraView.material = poseNet.transformMat;
        return true;
    }
}
