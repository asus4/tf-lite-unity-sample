using System.Threading;
using Cysharp.Threading.Tasks;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;
using TextureSource;

[RequireComponent(typeof(VirtualTextureSource))]
public class EfficientDetSample : MonoBehaviour
{
    [SerializeField]
    private EfficientDet.Options options = default;

    [SerializeField]
    private AspectRatioFitter frameContainer = null;

    [SerializeField]
    private Text framePrefab = null;

    [SerializeField, Range(0f, 1f)]
    private float scoreThreshold = 0.5f;

    [SerializeField]
    private TextAsset labelMap = null;

    [SerializeField]
    private bool runBackground = false;

    private EfficientDet efficientDet;
    private Text[] frames;
    private string[] labels;

    private UniTask<(bool IsCanceled, bool Result)> task;

    private void Start()
    {
        efficientDet = new EfficientDet(options);

        // Init frames
        frames = new Text[10];
        Transform parent = frameContainer.transform;
        for (int i = 0; i < frames.Length; i++)
        {
            frames[i] = Instantiate(framePrefab, Vector3.zero, Quaternion.identity, parent);
            frames[i].transform.localPosition = Vector3.zero;
        }

        // Labels
        labels = labelMap.text.Split('\n');

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
        efficientDet?.Dispose();
    }

    private void OnTextureUpdate(Texture texture)
    {
        if (runBackground)
        {
            if (task.Status.IsCompleted())
            {
                task = InvokeAsync(texture, destroyCancellationToken).SuppressCancellationThrow();
            }
        }
        else
        {
            Invoke(texture);
        }
    }

    private void Invoke(Texture texture)
    {
        efficientDet.Run(texture);
        UpdateResults(texture);
    }

    private async UniTask<bool> InvokeAsync(Texture texture, CancellationToken cancellationToken)
    {
        await efficientDet.RunAsync(texture, cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();
        UpdateResults(texture);
        return true;
    }

    private void UpdateResults(Texture texture)
    {
        var results = efficientDet.GetResults();
        Vector2 size = (frameContainer.transform as RectTransform).rect.size;

        float aspect = (float)texture.width / texture.height;
        Vector2 ratio = aspect > 1
            ? new Vector2(1.0f, 1 / aspect)
            : new Vector2(aspect, 1.0f);

        for (int i = 0; i < 10; i++)
        {
            SetFrame(frames[i], results[i], size * ratio);
        }
    }

    private void SetFrame(Text frame, EfficientDet.Result result, Vector2 size)
    {
        if (result.score < scoreThreshold)
        {
            frame.gameObject.SetActive(false);
            return;
        }
        else
        {
            frame.gameObject.SetActive(true);
        }

        frame.text = $"{GetLabelName(result.classID)} : {(int)(result.score * 100)}%";
        var rt = frame.transform as RectTransform;
        rt.anchoredPosition = result.rect.position * size - size * 0.5f;
        rt.sizeDelta = result.rect.size * size;
    }

    private string GetLabelName(int id)
    {
        if (id < 0 || id >= labels.Length - 1)
        {
            return "?";
        }
        return labels[id + 1];
    }

}
