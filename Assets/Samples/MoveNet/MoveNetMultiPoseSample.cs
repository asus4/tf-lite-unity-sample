using UnityEngine;
using TensorFlowLite;

[RequireComponent(typeof(WebCamInput))]
public class MoveNetMultiPoseSample : MonoBehaviour
{
    [SerializeField]
    MoveNetMultiPose.Options options = default;

    [SerializeField]
    private RectTransform cameraView = null;

    [SerializeField, Range(0, 1)]
    private float threshold = 0.3f;

    private MoveNetMultiPose moveNet;
    private readonly Vector3[] rtCorners = new Vector3[4];
    private PrimitiveDraw draw;
    private MoveNetPose[] poses;

    private void Start()
    {
        moveNet = new MoveNetMultiPose(options);
        draw = new PrimitiveDraw(Camera.main, gameObject.layer)
        {
            color = Color.green,
        };

        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.AddListener(OnTextureUpdate);
    }

    private void OnDestroy()
    {
        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.RemoveListener(OnTextureUpdate);
        moveNet?.Dispose();
        draw?.Dispose();
    }

    private void Update()
    {
        if (poses != null)
        {
            foreach (var pose in poses)
            {
                DrawPose(pose);
            }
        }
    }

    private void OnTextureUpdate(Texture texture)
    {
        Invoke(texture);
    }

    private void Invoke(Texture texture)
    {
        moveNet.Invoke(texture);
        poses = moveNet.GetResults();
    }

    private void DrawPose(MoveNetPose pose)
    {
        if (pose == null)
        {
            return;
        }

        cameraView.GetWorldCorners(rtCorners);
        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];

        var connections = PoseNet.Connections;
        int len = connections.GetLength(0);
        for (int i = 0; i < len; i++)
        {
            var a = pose[(int)connections[i, 0]];
            var b = pose[(int)connections[i, 1]];
            if (a.confidence >= threshold && b.confidence >= threshold)
            {
                draw.Line3D(
                    MathTF.Lerp(min, max, new Vector3(a.x, 1f - a.y, 0)),
                    MathTF.Lerp(min, max, new Vector3(b.x, 1f - b.y, 0)),
                    1
                );
            }
        }

        draw.Apply();
    }
}
