using System.Threading;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;
using Cysharp.Threading.Tasks;

[RequireComponent(typeof(WebCamInput))]
public class MoveNetSinglePoseSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")]
    private string fileName = default;

    [SerializeField]
    private RawImage cameraView = null;

    [SerializeField, Range(0, 1)]
    private float threshold = 0.3f;

    private MoveNet moveNet;
    private readonly Vector3[] rtCorners = new Vector3[4];
    private PrimitiveDraw draw;
    private MoveNet.Result[] results;

    private void Start()
    {
        moveNet = new MoveNet(fileName);
        draw = new PrimitiveDraw(Camera.main, gameObject.layer)
        {
            color = Color.green,
        };

        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.AddListener(OnTextureUpdate);
    }

    private void OnDestroy()
    {
        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.RemoveListener(OnTextureUpdate);
        moveNet?.Dispose();
        draw?.Dispose();
    }

    private void Update()
    {
        DrawResult(results);
    }

    private void OnTextureUpdate(Texture texture)
    {
        Invoke(texture);
    }

    private void Invoke(Texture texture)
    {
        moveNet.Invoke(texture);
        results = moveNet.GetResults();
        cameraView.material = moveNet.transformMat;
    }

    private void DrawResult(MoveNet.Result[] results)
    {
        if (results == null || results.Length == 0)
        {
            return;
        }

        var rect = cameraView.GetComponent<RectTransform>();
        rect.GetWorldCorners(rtCorners);
        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];

        var connections = PoseNet.Connections;
        int len = connections.GetLength(0);
        for (int i = 0; i < len; i++)
        {
            var a = results[(int)connections[i, 0]];
            var b = results[(int)connections[i, 1]];
            if (a.confidence >= threshold && b.confidence >= threshold)
            {
                draw.Line3D(
                    MathTF.Lerp(min, max, new Vector3(a.x, 1f - a.y, 0)),
                    MathTF.Lerp(min, max, new Vector3(b.x, 1f - b.y, 0)),
                    1
                );
            }
        }

        draw.Apply();
    }
}
