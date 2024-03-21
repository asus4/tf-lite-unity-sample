using System.Collections.Generic;
using System.Linq;
using TensorFlowLite;
using TextureSource;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

/// <summary>
/// MagicTouch model from MediaPipe's interactive segmentation task
/// https://developers.google.com/mediapipe/solutions/vision/interactive_segmenter
/// 
/// Licensed under Apache License 2.0
/// See model card for details
/// https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MagicTouch.pdf
/// </summary>
[RequireComponent(typeof(VirtualTextureSource))]
public sealed class MagicTouchSample : MonoBehaviour
{
    [Header("Configurations")]
    [SerializeField, FilePopup("*.tflite")]
    private string modelFile = string.Empty;

    [SerializeField]
    private MagicTouch.Options options = default;

    [Header("UI")]
    [SerializeField]
    private RawImage outputView = null;

    [SerializeField]
    private RawImage debugPreview = null;

    [SerializeField]
    private RectTransform positivePointPrefab = null;

    private MagicTouch magicTouch;
    private Texture inputTexture;


    private readonly List<RectTransform> positivePoints = new();

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

    public void OnClick(BaseEventData baseEventData)
    {
        if (baseEventData is not PointerEventData pointerEventData)
        {
            Debug.LogWarning($"Unknown event: {baseEventData}", this);
            return;
        }

        // Create marker at clicked position
        var container = outputView.rectTransform;
        var point = Instantiate(positivePointPrefab, container);
        Vector2 screenPos = pointerEventData.position;
        Vector2 normalizedPos = new(screenPos.x / Screen.width, screenPos.y / Screen.height);
        point.anchoredPosition = (normalizedPos - container.pivot) * container.rect.size;
        positivePoints.Add(point);

        // Run
        Run();
    }

    public void RemoveAllPoints()
    {
        foreach (var point in positivePoints)
        {
            Destroy(point.gameObject);
        }
        positivePoints.Clear();
    }

    private void Run()
    {
        if (inputTexture == null)
        {
            return;
        }

        var container = outputView.rectTransform;
        var points = positivePoints.Select(point =>
        {
            // convert to 0-1 range
            return point.anchoredPosition / container.rect.size + container.pivot;
        });
        magicTouch.SetPoints(points);
        magicTouch.Run(inputTexture);

        debugPreview.texture = magicTouch.DebugInputTexture;
    }
}
