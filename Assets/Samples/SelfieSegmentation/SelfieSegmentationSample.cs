using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;
using TextureSource;

[RequireComponent(typeof(VirtualTextureSource))]
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
        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.AddListener(OnTextureUpdate);
        }
    }

    private void OnDestroy()
    {
        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.RemoveListener(OnTextureUpdate);
        }
        segmentation?.Dispose();
    }

    private void OnTextureUpdate(Texture texture)
    {
        segmentation.Run(texture);
        outputView.texture = segmentation.GetResultTexture();
    }
}
