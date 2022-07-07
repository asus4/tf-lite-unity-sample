using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;
using TensorFlowLite;
using TensorFlowLite.MoveNet;

[RequireComponent(typeof(WebCamInput))]
public class MoveNetMultiPoseSample : MonoBehaviour
{
    [SerializeField]
    MoveNetMultiPose.Options options = default;

    [SerializeField]
    private RectTransform cameraView = null;

    [SerializeField]
    private bool runBackground = false;

    [SerializeField, Range(0, 1)]
    private float threshold = 0.3f;

    private MoveNetMultiPose moveNet;
    private MoveNetPoseWithBoundingBox[] poses;
    private MoveNetDrawer drawer;

    private UniTask<bool> task;
    private CancellationToken cancellationToken;

    private void Start()
    {
        moveNet = new MoveNetMultiPose(options);
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
        if (poses != null)
        {
            foreach (var pose in poses)
            {
                drawer.DrawPose(pose, threshold);
            }
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
        poses = moveNet.GetResults();
    }

    private async UniTask<bool> InvokeAsync(Texture texture)
    {
        await moveNet.InvokeAsync(texture, cancellationToken);
        poses = moveNet.GetResults();
        return true;
    }
}
