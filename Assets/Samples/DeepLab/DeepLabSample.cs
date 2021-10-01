using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(WebCamInput))]
public class DeepLabSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")]
    private string fileName = "deeplabv3_257_mv_gpu.tflite";

    [SerializeField]
    private RawImage cameraView = null;

    [SerializeField]
    private RawImage outputView = null;

    [SerializeField]
    private ComputeShader compute = null;

    private DeepLab deepLab;

    private void Start()
    {
        deepLab = new DeepLab(fileName, compute);

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

        // Slow but works on mobile
        // outputView.texture = deepLab.GetResultTexture2D();

        // Fast but errors on mobile. Need to be fixed 
        outputView.texture = deepLab.GetResultTexture();

    }
}
