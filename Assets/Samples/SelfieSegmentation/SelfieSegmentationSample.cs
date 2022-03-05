using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(WebCamInput))]
public class SelfieSegmentationSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")]
    private string fileName = "deeplabv3_257_mv_gpu.tflite";

    [SerializeField]
    private RawImage cameraView = null;

    [SerializeField]
    private RawImage outputView = null;

    [SerializeField]
    private ComputeShader compute = null;

    private SelfieSegmentation segmentation;

    private void Start()
    {
        segmentation = new SelfieSegmentation(fileName, compute);

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
        cameraView.material = segmentation.transformMat;
        outputView.texture = segmentation.GetResultTexture();
    }
}
