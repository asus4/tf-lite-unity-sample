using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;

public class TextClassificationSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "deeplabv3_257_mv_gpu.tflite";
    [SerializeField] TextAsset vocabularyText = null;
    [SerializeField] InputField textInput = null;
    [SerializeField] Text resultLabel = null;

    TextClassification textClassification;

    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        textClassification = new TextClassification(path, vocabularyText.text);

        OnTextChanged(textInput.text);
    }

    void OnDestroy()
    {
        textClassification?.Dispose();
    }

    public void OnTextChanged(string text)
    {
        Debug.Log(text);
        textClassification.Invoke(text);
        var result = textClassification.result;
        resultLabel.text = $"Positive: {(result.positive * 100):0.0}%\n"
                         + $"Negative: { (result.negative * 100):0.0}%";
    }
}
