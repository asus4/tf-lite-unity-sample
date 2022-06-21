using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;
using TensorFlowLite;
using TensorFlowLite.MoveNet;

[RequireComponent(typeof(WebCamInput))]
public class MoveNetSinglePoseSample : MonoBehaviour
{
    [SerializeField]
    MoveNetSinglePose.Options options = default;

    [SerializeField]
    private RectTransform cameraView = null;

    [SerializeField]
    private bool runBackground = false;

    [SerializeField, Range(0, 1)]
    private float threshold = 0.3f;

    private MoveNetSinglePose moveNet;
    private MoveNetPose pose;
    private MoveNetDrawer drawer;

    private UniTask<bool> task;
    private CancellationToken cancellationToken;

    private void Start()
    {
        moveNet = new MoveNetSinglePose(options);
        drawer = new MoveNetDrawer(Camera.main, cameraView);

        cancellationToken = this.GetCancellationTokenOnDestroy();

        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.AddListener(OnTextureUpdate);
    }

    private void OnDestroy()
    {
        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.RemoveListener(OnTextureUpdate);
        moveNet?.Dispose();
        drawer?.Dispose();
    }

    private void Update()
    {
        if (pose != null)
        {
            drawer.DrawPose(pose, threshold);
        }
    }

    private void OnTextureUpdate(Texture texture)
    {
        if (runBackground)
        {
            if (task.Status.IsCompleted())
            {
                task = InvokeAsync(texture);
            }
        }
        else
        {
            Invoke(texture);
        }
    }

    private void Invoke(Texture texture)
    {
        moveNet.Invoke(texture);
        pose = moveNet.GetResult();
    }

    private async UniTask<bool> InvokeAsync(Texture texture)
    {
        await moveNet.InvokeAsync(texture, cancellationToken);
        pose = moveNet.GetResult();
        return true;
    }
}
