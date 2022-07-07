using System.Linq;
using System.Text;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.Scripting;
using UnityEngine.UI;
using TextureSource;

/// <summary>
/// MoViNets: Video Classification example from TensorFlow
/// https://www.tensorflow.org/lite/examples/video_classification/overview
/// </summary>
[RequireComponent(typeof(VirtualTextureSource))]
public class VideoClassificationSample : MonoBehaviour
{
    [SerializeField]
    private Text resultLabel = null;

    [SerializeField]
    private VideoClassification.Options options = default;

    [SerializeField]
    [Range(1, 10)]
    private int resultCount = 3;

    private VideoClassification classification;
    private readonly StringBuilder sb = new StringBuilder();

    private void Start()
    {
        classification = new VideoClassification(options);
        GetComponent<VirtualTextureSource>().OnTexture.AddListener(Invoke);
    }

    private void OnDestroy()
    {
        GetComponent<VirtualTextureSource>().OnTexture.RemoveListener(Invoke);
        classification?.Dispose();
    }

    private void Invoke(Texture texture)
    {
        classification.Invoke(texture);

        var categories = classification.GetResults().Take(resultCount);

        sb.Clear();
        sb.AppendLine($"TOP {resultCount} CLASSES:");
        foreach (var category in categories)
        {
            string label = classification.GetLabel(category.label);
            sb.AppendLine($"{(int)(category.score * 100f)}% : {label}");
        }
        resultLabel.text = sb.ToString();
    }

    // Called by button via UnityEvent
    [Preserve]
    public void ResetStates()
    {
        Debug.Log("Reset states");
        classification.ResetStates();
    }
}
