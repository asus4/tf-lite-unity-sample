using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;

public class DeepLabSample : MonoBehaviour
{
    [SerializeField] string fileName = "deeplabv3_257_mv_gpu.tflite";
    [SerializeField] RawImage cameraView = null;
    [SerializeField] RawImage outputView = null;

    WebCamTexture webcamTexture;
    DeepLab deepLab;


    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        deepLab = new DeepLab(path);

        // Init camera
        string cameraName = WebCamUtil.FindName();
        webcamTexture = new WebCamTexture(cameraName, 640, 480, 30);
        webcamTexture.Play();
        cameraView.texture = webcamTexture;

    }

    void OnDestroy()
    {
        webcamTexture?.Stop();
        deepLab?.Dispose();
    }

    void Update()
    {
        deepLab.Invoke(webcamTexture);
        outputView.texture = deepLab.GetResultTexture();

        cameraView.uvRect = TextureToTensor.GetUVRect(
            (float)webcamTexture.width / webcamTexture.height,
            1,
            TextureToTensor.AspectMode.Fill);
    }

}
