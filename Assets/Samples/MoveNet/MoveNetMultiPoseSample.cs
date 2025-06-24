using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;
using TensorFlowLite.MoveNet;
using TextureSource;

[RequireComponent(typeof(VirtualTextureSource))]
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

    private void Start()
    {
        moveNet = new MoveNetMultiPose(options);
        drawer = new MoveNetDrawer(Camera.main, cameraView);

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
                task = InvokeAsync(texture, destroyCancellationToken);
            }
        }
        else
        {
            Invoke(texture);
        }
    }

    private void Invoke(Texture texture)
    {
        moveNet.Run(texture);
        poses = moveNet.GetResults();
    }

    private async UniTask<bool> InvokeAsync(Texture texture, CancellationToken cancellationToken)
    {
        await moveNet.RunAsync(texture, cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();
        poses = moveNet.GetResults();
        return true;
    }
}
