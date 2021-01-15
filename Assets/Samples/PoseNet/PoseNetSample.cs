using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;
using Cysharp.Threading.Tasks;

public class PoseNetSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite";
    [SerializeField] RawImage cameraView = null;
    [SerializeField, Range(0f, 1f)] float threshold = 0.5f;
    [SerializeField, Range(0f, 1f)] float lineThickness = 0.5f;
    [SerializeField] bool runBackground;

    WebCamTexture webcamTexture;
    PoseNet poseNet;
    Vector3[] corners = new Vector3[4];
    PrimitiveDraw draw;
    UniTask<bool> task;
    PoseNet.Result[] results;
    CancellationToken cancellationToken;

    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        poseNet = new PoseNet(path);

        // Init camera
        string cameraName = WebCamUtil.FindName();
        webcamTexture = new WebCamTexture(cameraName, 640, 480, 30);
        webcamTexture.Play();
        cameraView.texture = webcamTexture;

        draw = new PrimitiveDraw()
        {
            color = Color.green,
        };

        cancellationToken = this.GetCancellationTokenOnDestroy();
    }

    void OnDestroy()
    {
        webcamTexture?.Stop();
        poseNet?.Dispose();
        draw?.Dispose();
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
            poseNet.Invoke(webcamTexture);
            results = poseNet.GetResults();
            cameraView.material = poseNet.transformMat;
        }

        if (results != null)
        {
            DrawResult();
        }
    }

    void DrawResult()
    {
        var rect = cameraView.GetComponent<RectTransform>();
        rect.GetWorldCorners(corners);
        Vector3 min = corners[0];
        Vector3 max = corners[2];

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
                    lineThickness
                );
            }
        }

        draw.Apply();
    }

    async UniTask<bool> InvokeAsync()
    {
        results = await poseNet.InvokeAsync(webcamTexture, cancellationToken);
        cameraView.material = poseNet.transformMat;
        return true;
    }
}
