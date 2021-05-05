using System.Collections;
using System.Collections.Generic;
using System.IO;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

public class MeetSegmentationSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "deeplabv3_257_mv_gpu.tflite";
    [SerializeField] RawImage cameraView = null;
    [SerializeField] RawImage outputView = null;
    [SerializeField] ComputeShader compute = null;

    WebCamTexture webcamTexture;
    MeetSegmentation segmentation;


    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        segmentation = new MeetSegmentation(path, compute);

        // Init camera
        string cameraName = WebCamUtil.FindName();
        webcamTexture = new WebCamTexture(cameraName, 1280, 720, 30);
        webcamTexture.Play();
        cameraView.texture = webcamTexture;

    }

    void OnDestroy()
    {
        webcamTexture?.Stop();
        segmentation?.Dispose();
    }

    void Update()
    {
        segmentation.Invoke(webcamTexture);
        cameraView.material = segmentation.transformMat;

        outputView.texture = segmentation.GetResultTexture();
    }
}
