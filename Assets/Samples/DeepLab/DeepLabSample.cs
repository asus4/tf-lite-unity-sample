using System.Collections;
using System.Collections.Generic;
using System.IO;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

public class DeepLabSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "deeplabv3_257_mv_gpu.tflite";
    [SerializeField] RawImage cameraView = null;
    [SerializeField] RawImage outputView = null;
    [SerializeField] ComputeShader compute = null;

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
        deepLab.Invoke(webcamTexture);
        // Slow but works on mobile
        outputView.texture = deepLab.GetResultTexture2D();

        // Fast but errors on mobile. Need to be fixed 
        // outputView.texture = deepLab.GetResultTexture();

        cameraView.material = deepLab.transformMat;
    }

}
