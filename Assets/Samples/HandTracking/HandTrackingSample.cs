using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;

public class HandTrackingSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string palmModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField] TextAsset anchorCsv = null;
    [SerializeField] RawImage cameraView = null;
    [SerializeField] Image framePrefab = null;

    WebCamTexture webcamTexture;
    PalmDetect palmDetect;

    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, palmModelFile);
        palmDetect = new PalmDetect(path, anchorCsv.text);

        string cameraName = WebCamUtil.FindName();
        webcamTexture = new WebCamTexture(cameraName, 1280, 720, 30);
        cameraView.texture = webcamTexture;
        webcamTexture.Play();
        Debug.Log($"Starting camera: {cameraName}");
    }
    void OnDestroy()
    {
        webcamTexture?.Stop();
        palmDetect?.Dispose();
    }

    void Update()
    {
        var resizeOptions = palmDetect.ResizeOptions;
        resizeOptions.rotationDegree = webcamTexture.videoRotationAngle;
        palmDetect.ResizeOptions = resizeOptions;

        palmDetect.Invoke(webcamTexture);

        cameraView.material = palmDetect.transformMat;

        var palms = palmDetect.GetResults(0.7f);
        Debug.Log(palms.Count);
    }



}
