using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(WebCamInput))]
public class SelfieSegmentationSample : MonoBehaviour
{
    [SerializeField]
    private RawImage outputView = null;

    [SerializeField]
    private SelfieSegmentation.Options options = default;

    private SelfieSegmentation segmentation;

    private void Start()
    {
        segmentation = new SelfieSegmentation(options);

        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.AddListener(OnTextureUpdate);
    }

    private void OnDestroy()
    {
        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.RemoveListener(OnTextureUpdate);

        segmentation?.Dispose();
    }

    private void OnTextureUpdate(Texture texture)
    {
        segmentation.Invoke(texture);
        outputView.texture = segmentation.GetResultTexture();
    }
}
