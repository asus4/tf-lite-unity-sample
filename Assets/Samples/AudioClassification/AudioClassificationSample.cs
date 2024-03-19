using System.Collections;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

public sealed class AudioClassificationSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")]
    private string modelFile = string.Empty;

    [SerializeField]
    private TextAsset labels;

    [SerializeField]
    private Text resultText = null;

    [SerializeField]
    private MicrophoneBuffer.Options microphoneOptions = new();

    private AudioClassification classification;
    private MicrophoneBuffer mic;

    private IEnumerator Start()
    {
        classification = new AudioClassification(FileUtil.LoadFile(modelFile));

        mic = new MicrophoneBuffer();
        yield return mic.StartRecording(microphoneOptions);
    }

    private void OnDestroy()
    {
        classification?.Dispose();
        mic?.Dispose();
    }

    private void Run()
    {
        if (!mic.IsRecording) return;

        mic.GetLatestSamples(classification.Input);
        classification.Run();
    }
}
