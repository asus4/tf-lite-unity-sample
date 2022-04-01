using System.Threading;
using UnityEngine;
using TensorFlowLite;
using Cysharp.Threading.Tasks;

[RequireComponent(typeof(WebCamInput))]
public class MoveNetMultiplePoseSample : MonoBehaviour
{
    [SerializeField]
    MoveNetMultiple.Options options = default;

    [SerializeField]
    private RectTransform cameraView = null;

    [SerializeField, Range(0, 1)]
    private float threshold = 0.3f;

    private MoveNetMultiple moveNet;
    private readonly Vector3[] rtCorners = new Vector3[4];
    private PrimitiveDraw draw;
    // private MoveNet.Result[] results;
    private MoveNetMultiple.Pose[] poses;

    private void Start()
    {
        moveNet = new MoveNetMultiple(options);
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
        // DrawResult(results);
        // Debug.Log($"poses.Length: {poses.Length}");
        DrawResult(poses);
    }

    private void OnTextureUpdate(Texture texture)
    {
        Invoke(texture);
    }

    private void Invoke(Texture texture)
    {
        moveNet.Invoke(texture);
        // results = moveNet.GetResults();
        poses = moveNet.GetPoses();
    }

    // private void DrawResult(MoveNet.Result[] results)
    // {
    //     if (results == null || results.Length == 0)
    //     {
    //         return;
    //     }

    //     cameraView.GetWorldCorners(rtCorners);
    //     Vector3 min = rtCorners[0];
    //     Vector3 max = rtCorners[2];

    //     var connections = PoseNet.Connections;
    //     int len = connections.GetLength(0);
    //     for (int i = 0; i < len; i++)
    //     {
    //         var a = results[(int)connections[i, 0]];
    //         var b = results[(int)connections[i, 1]];
    //         if (a.confidence >= threshold && b.confidence >= threshold)
    //         {
    //             draw.Line3D(
    //                 MathTF.Lerp(min, max, new Vector3(a.x, 1f - a.y, 0)),
    //                 MathTF.Lerp(min, max, new Vector3(b.x, 1f - b.y, 0)),
    //                 1
    //             );
    //         }
    //     }

    //     draw.Apply();
    // }
    private async void DrawResult(MoveNetMultiple.Pose[] poses)
    {
        if (poses == null || poses.Length == 0)
        {
            Debug.Log("poses is null or empty");
            return;
        }
        cameraView.GetWorldCorners(rtCorners);
        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];
        var connections = MoveNet.Connections;
        int len = connections.GetLength(0);
        int MultiPoseInstanceSize = 17 * 3 + 5;
        var numInstances = 6;
        for (int j = 0; j < numInstances; j++)
        {
            Debug.Log(poses[j].score);
            // if (poses[j].score < threshold)
            // {
            //     continue;
            // }
            for (int i = 0; i < len; i++)
            {
                var a = poses[j].keypoints[(int)connections[i, 0]];
                var b = poses[j].keypoints[(int)connections[i, 1]];
                if (a.confidence >= threshold && b.confidence >= threshold)
                {
                    draw.Line3D(
                        MathTF.Lerp(min, max, new Vector3(a.x, 1f - a.y, 0)),
                        MathTF.Lerp(min, max, new Vector3(b.x, 1f - b.y, 0)),
                        1
                    );
                }
            }
        }
        draw.Apply();
    }
}
