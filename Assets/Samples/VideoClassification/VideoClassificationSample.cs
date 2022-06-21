using System.Collections;
using System.Collections.Generic;
using TensorFlowLite;
using UnityEngine;

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
