using System.Collections;
using System.Text;
using TensorFlowLite;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// TensorFlow Lite Audio Classification
/// https://www.tensorflow.org/lite/examples/audio_classification/overview
/// https://www.kaggle.com/models/google/yamnet/frameworks/tensorFlow2/variations/yamnet/versions/1?tfhub-redirect=true
/// </summary>
public sealed class AudioClassificationSample : MonoBehaviour
{
    [Header("Configs")]
    [SerializeField, FilePopup("*.tflite")]
    private string modelFile = string.Empty;

    [SerializeField]
    private TextAsset labelFile;

    [SerializeField]
    [Range(0.1f, 5f)]
    private float runEachNSec = 0.2f;


    [SerializeField]
    [Range(1, 10)]
    private int showTopKLabels = 3;

    [SerializeField]
    private MicrophoneBuffer.Options microphoneOptions = new();

    [Header("UI")]
    [SerializeField]
    private Text resultText = null;

    [SerializeField]
    private RectTransform waveformView = null;


    private AudioClassification classification;
    private MicrophoneBuffer mic;
    private string[] labelNames;
    private PrimitiveDraw waveFormDrawer;
    private readonly Vector3[] rtCorners = new Vector3[4];
    private readonly StringBuilder sb = new();

    private IEnumerator Start()
    {
        labelNames = labelFile.text.Split('\n');
        classification = new AudioClassification(FileUtil.LoadFile(modelFile));
        waveFormDrawer = new PrimitiveDraw(Camera.main, gameObject.layer);

        mic = new MicrophoneBuffer();
        yield return mic.StartRecording(microphoneOptions);

        while (Application.isPlaying)
        {
            yield return new WaitForSeconds(runEachNSec);
            Run();
        }
    }

    private void OnDestroy()
    {
        classification?.Dispose();
        mic?.Dispose();
    }

    private void Update()
    {
        waveFormDrawer.Apply(drawEditor: true, clear: false);
    }

    private void Run()
    {
        if (!mic.IsRecording) return;

        mic.GetLatestSamples(classification.Input);
        classification.Run();

        sb.Clear();
        sb.AppendLine($"Top {showTopKLabels}:");
        var labels = classification.GetTopLabels(showTopKLabels);
        for (int i = 0; i < labels.Length; i++)
        {
            sb.AppendLine($"{labelNames[labels[i].id]}: {labels[i].score * 100f:F1}%");
        }
        resultText.text = sb.ToString();

        UpdateWaveform(classification.Input);
    }

    // TODO: optimize this
    private void UpdateWaveform(float[] input)
    {
        waveformView.GetWorldCorners(rtCorners);
        float3 min = rtCorners[0];
        float3 max = rtCorners[2];

        waveFormDrawer.Clear();
        waveFormDrawer.color = Color.green;

        int length = input.Length;
        float delta = 1f / length;

        Vector3 prev = math.lerp(min, max, new float3(0, input[0] * 0.5f + 0.5f, 0));

        const int STRIDE = 8;
        for (int i = 1; i < length; i += STRIDE)
        {
            float3 t = new(i * delta, input[i] * 0.5f + 0.5f, 0);
            Vector3 point = math.lerp(min, max, t);
            waveFormDrawer.Line(prev, point, 0.01f);
            prev = point;
        }
    }
}
