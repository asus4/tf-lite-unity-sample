using TensorFlowLite;
using TextureSource;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(VirtualTextureSource))]
public class DeepLabSample : MonoBehaviour
{
    [SerializeField]
    private RawImage outputView = null;

    [SerializeField]
    private DeepLab.Options options = default;

    private DeepLab deepLab;

    private void Start()
    {
        deepLab = new DeepLab(options);
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
        deepLab?.Dispose();
    }

    private void OnTextureUpdate(Texture texture)
    {
        deepLab.Run(texture);
        outputView.texture = deepLab.GetResultTexture();
    }
}
