using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using TensorFlowLite;

public class SmartReplySample : MonoBehaviour
{
    [SerializeField] string fileName = "deeplabv3_257_mv_gpu.tflite";

    SmartReply smartReply;

    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        smartReply = new SmartReply(path);
    }

    void OnDestroy()
    {
        smartReply?.Dispose();
    }

}
