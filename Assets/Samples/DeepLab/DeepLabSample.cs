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
    [SerializeField] ComputeShader compute = null;
    [SerializeField] Texture2D inputTex;

    WebCamTexture webcamTexture;
    DeepLab deepLab;


    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        deepLab = new DeepLab(path, compute);

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
        if (inputTex != null)
        {
            Invoke(inputTex);
        }
        else
        {
            Invoke(webcamTexture);
        }
    }

    void Invoke(Texture texture)
    {
        deepLab.Invoke(texture);
        cameraView.texture = texture;
        outputView.texture = deepLab.GetResultTexture2D();
        float aspect = (float)texture.width / (float)texture.height;
        cameraView.uvRect = TextureToTensor.GetUVRect(aspect, 1, TextureToTensor.AspectMode.Fill);
    }

}
