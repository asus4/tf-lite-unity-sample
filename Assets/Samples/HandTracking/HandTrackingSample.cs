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

    Image[] frames;

    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, palmModelFile);
        palmDetect = new PalmDetect(path, anchorCsv.text);

        string cameraName = WebCamUtil.FindName();
        webcamTexture = new WebCamTexture(cameraName, 1280, 720, 30);
        cameraView.texture = webcamTexture;
        webcamTexture.Play();
        Debug.Log($"Starting camera: {cameraName}");

        // Init frames
        frames = new Image[PalmDetect.MAX_PALM_NUM];
        var parent = cameraView.transform;
        for (int i = 0; i < frames.Length; i++)
        {
            frames[i] = Instantiate(framePrefab, Vector3.zero, Quaternion.identity, parent);
        }
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

        var palms = palmDetect.GetResults(0.7f, 0.3f);
        var size = ((RectTransform)cameraView.transform).rect.size;
        for (int i = 0; i < palms.Count; i++)
        {
            frames[i].gameObject.SetActive(true);
            SetFrame(frames[i], palms[i], size);
        }
        for (int i = palms.Count; i < frames.Length; i++)
        {
            frames[i].gameObject.SetActive(false);
        }
    }

    void SetFrame(Graphic frame, PalmDetect.Palm palm, Vector2 size)
    {
        var rt = frame.transform as RectTransform;
        var p = palm.rect.position;
        p.y = 1.0f - p.y; // invert Y
        rt.anchoredPosition = p * size - size * 0.5f;
        rt.sizeDelta = palm.rect.size * size;
    }

}
