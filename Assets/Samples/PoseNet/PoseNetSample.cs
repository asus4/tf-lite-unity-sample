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

    WebCamTexture webcamTexture;
    PoseNet poseNet;

    public float[] heatmap;
    public float[] offsets;
    public PoseNet.Result[] results;

    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        poseNet = new PoseNet(path);

        // Init camera
        string cameraName = GetWebcamName();
        webcamTexture = new WebCamTexture(cameraName, 1280, 720);
        cameraView.texture = webcamTexture;
        webcamTexture.Play();
        Debug.Log($"Starting camera: {cameraName}");
    }

    void OnDestroy()
    {
        webcamTexture?.Stop();
        poseNet?.Dispose();
    }

    void Update()
    {
        poseNet.Invoke(webcamTexture);
        heatmap = poseNet.heatmap;
        offsets = poseNet.offsets;
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
}
