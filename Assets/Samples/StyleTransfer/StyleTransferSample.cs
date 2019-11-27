using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;

public class StyleTransferSample : MonoBehaviour
{
    [SerializeField] string predictionFileName = "style_predict_quantized_256.tflite";
    [SerializeField] string transferFileName = "style_transfer_quantized_dynamic.tflite";
    [SerializeField] Texture2D styleImage = null;
    [SerializeField] RawImage preview = null;

    WebCamTexture webcamTexture;
    StyleTransfer styleTransfer;
    public float[] styleBottleneck;

    void Start()
    {
        // Predict style bottleneck;
        string predictionModelPath = Path.Combine(Application.streamingAssetsPath, predictionFileName);
        using (var predict = new StylePredict(predictionModelPath))
        {
            predict.Invoke(styleImage);
            styleBottleneck = predict.GetStyleBottleneck();
        }

        

        // Init camera
        string cameraName = WebCamUtil.FindName();
        webcamTexture = new WebCamTexture(cameraName, 640, 480, 30);
        webcamTexture.Play();
        preview.texture = webcamTexture;

    }

    void OnDestroy()
    {
        styleTransfer?.Dispose();
    }

    void Update()
    {

        preview.uvRect = TextureToTensor.GetUVRect(
            (float)webcamTexture.width / (float)webcamTexture.height,
            1,
            TextureToTensor.AspectMode.Fill);
    }
}
