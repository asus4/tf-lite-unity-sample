using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;
using TextureSource;

/// <summary>
/// MagicTouch model from MediaPipe's interactive segmentation task
/// https://developers.google.com/mediapipe/solutions/vision/interactive_segmenter
/// 
/// Licensed under Apache License 2.0
/// See model card for details
/// https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MagicTouch.pdf
/// </summary>
[RequireComponent(typeof(VirtualTextureSource))]
public class MagicTouchSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")]
    private string modelFile = string.Empty;

    [SerializeField]
    private MagicTouch.Options options = default;

    [SerializeField]
    private RawImage outputView = null;

    private MagicTouch magicTouch;
    private Texture inputTexture;

    private void Start()
    {
        magicTouch = new MagicTouch(modelFile, options);
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
        magicTouch?.Dispose();
    }

    private void OnTextureUpdate(Texture texture)
    {
        inputTexture = texture;
    }
}
