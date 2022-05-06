using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(WebCamInput))]
public class DeepLabSample : MonoBehaviour
{
    [SerializeField]
    private RawImage cameraView = null;

    [SerializeField]
    private RawImage outputView = null;

    [SerializeField]
    private DeepLab.Options options = default;

    private DeepLab deepLab;

    private void Start()
    {
        deepLab = new DeepLab(options);

        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.AddListener(OnTextureUpdate);
    }

    private void OnDestroy()
    {
        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.RemoveListener(OnTextureUpdate);

        deepLab?.Dispose();
    }

    private void OnTextureUpdate(Texture texture)
    {
        deepLab.Invoke(texture);
        cameraView.material = deepLab.transformMat;
        outputView.texture = deepLab.GetResultTexture();
    }
}
