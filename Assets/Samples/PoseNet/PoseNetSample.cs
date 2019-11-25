using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;

public class PoseNetSample : MonoBehaviour
{
    [SerializeField] string fileName = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite";
    [SerializeField] RawImage cameraView = null;
    [SerializeField, Range(0f, 1f)] float threshold = 0.5f;

    WebCamTexture webcamTexture;
    PoseNet poseNet;

    public PoseNet.Result[] results;

    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        poseNet = new PoseNet(path);

        // Init camera
        string cameraName = GetWebcamName();
        webcamTexture = new WebCamTexture(cameraName, 1280, 720);
        webcamTexture.Play();
        cameraView.texture = webcamTexture;
    }

    void OnDestroy()
    {
        webcamTexture?.Stop();
        poseNet?.Dispose();
    }

    void Update()
    {
        poseNet.Invoke(webcamTexture);
        results = poseNet.GetResults();
    }

    static string GetWebcamName()
    {
        if (Application.isMobilePlatform)
        {
            return WebCamTexture.devices.Where(d => !d.isFrontFacing).Last().name;

        }
        return WebCamTexture.devices.Last().name;
    }


    void OnDrawGizmos()
    {
        if (results.Length == 0)
        {
            return;
        }

        float w = Screen.width;
        float h = Screen.height;

        Gizmos.color = Color.green;
        
        // Spheres
        foreach (var result in results)
        {
            if (result.confidence >= threshold)
            {
                var p = new Vector3(result.x * w, result.y * h, 0);
                Gizmos.DrawWireSphere(p, 20);
            }
        }

        // Lines
        var connections = PoseNet.Connections;
        int len = connections.GetLength(0);
        for (int i = 0; i < len; i++)
        {
            var a = results[(int)connections[i, 0]];
            var b = results[(int)connections[i, 1]];

            if (a.confidence >= threshold && b.confidence >= threshold)
            {
                Gizmos.DrawLine(new Vector3(a.x * w, a.y * h, 0),
                                new Vector3(b.x * w, b.y * h, 0));
            }
        }
    }

}
