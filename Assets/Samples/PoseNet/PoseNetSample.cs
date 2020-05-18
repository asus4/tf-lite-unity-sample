using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;

public class PoseNetSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite";
    [SerializeField] RawImage cameraView = null;
    [SerializeField] GLDrawer glDrawer = null;
    [SerializeField, Range(0f, 1f)] float threshold = 0.5f;

    WebCamTexture webcamTexture;
    PoseNet poseNet;
    Vector3[] corners = new Vector3[4];

    public PoseNet.Result[] results;

    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        poseNet = new PoseNet(path);

        // Init camera
        string cameraName = WebCamUtil.FindName();
        webcamTexture = new WebCamTexture(cameraName, 640, 480, 30);
        webcamTexture.Play();
        cameraView.texture = webcamTexture;

        glDrawer.OnDraw += OnGLDraw;
    }

    void OnDestroy()
    {
        webcamTexture?.Stop();
        poseNet?.Dispose();
        glDrawer.OnDraw -= OnGLDraw;
    }

    void Update()
    {
        poseNet.Invoke(webcamTexture);
        results = poseNet.GetResults();

        // set uv
        cameraView.material = poseNet.transformMat;
        // cameraView.uvRect = TextureToTensor.GetUVRect(
        //     (float)webcamTexture.width / webcamTexture.height,
        //     1,
        //     TextureToTensor.AspectMode.Fill);
    }

    void OnGLDraw()
    {
        var rect = cameraView.GetComponent<RectTransform>();
        rect.GetWorldCorners(corners);
        Vector3 min = corners[0];
        Vector3 max = corners[2];

        GL.Begin(GL.LINES);

        GL.Color(Color.green);
        var connections = PoseNet.Connections;
        int len = connections.GetLength(0);
        for (int i = 0; i < len; i++)
        {
            var a = results[(int)connections[i, 0]];
            var b = results[(int)connections[i, 1]];
            if (a.confidence >= threshold && b.confidence >= threshold)
            {
                GL.Vertex(Leap3(min, max, new Vector3(a.x, 1f - a.y, 0)));
                GL.Vertex(Leap3(min, max, new Vector3(b.x, 1f - b.y, 0)));
            }
        }

        GL.End();
    }

    static Vector3 Leap3(in Vector3 a, in Vector3 b, in Vector3 t)
    {
        return new Vector3(
            Mathf.Lerp(a.x, b.x, t.x),
            Mathf.Lerp(a.y, b.y, t.y),
            Mathf.Lerp(a.z, b.z, t.z)
        );
    }

}
