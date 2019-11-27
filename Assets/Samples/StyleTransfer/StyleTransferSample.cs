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

    public float[] styleBottleneck;

    void Start()
    {
        string predictionModelPath = Path.Combine(Application.streamingAssetsPath, predictionFileName);
        using (var predict = new StylePredict(predictionModelPath))
        {
            predict.Invoke(styleImage);
            styleBottleneck = predict.GetStyleBottleneck();
        }

    }

    void Update()
    {

    }
}
