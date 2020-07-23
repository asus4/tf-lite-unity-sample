using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using TensorFlowLite;

public class SmartReplySample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "smartreply.tflite";
    [SerializeField] TextAsset responseText = null;

    SmartReply smartReply;

    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        var responses  = responseText.text.Split('\n');
        smartReply = new SmartReply(path, responses);

        // TODO add ui
        smartReply.Invoke("How are you");

    }

    void OnDestroy()
    {
        smartReply?.Dispose();
    }



}
