using TensorFlowLite;
using UnityEngine;

/// <summary>
/// Video Classification example from TensorFlow
/// https://www.tensorflow.org/lite/examples/video_classification/overview
/// </summary>
[RequireComponent(typeof(WebCamInput))]
public class VideoClassificationSample : MonoBehaviour
{
    [SerializeField]
    VideoClassification.Options options = default;

    private VideoClassification classification;

    private void Start()
    {
        classification = new VideoClassification(options);
        GetComponent<WebCamInput>().OnTextureUpdate.AddListener(Invoke);
    }

    private void OnDestroy()
    {
        GetComponent<WebCamInput>().OnTextureUpdate.RemoveListener(Invoke);
        classification?.Dispose();
    }

    private void Invoke(Texture texture)
    {
        classification.Invoke(texture);
    }
}
