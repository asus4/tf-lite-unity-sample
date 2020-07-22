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
    [SerializeField, FilePopup("*.tflite")] string landmarkModelFile = "coco_ssd_mobilenet_quant.tflite";

    [SerializeField] RawImage cameraView = null;
    [SerializeField] Image framePrefab = null;

    WebCamTexture webcamTexture;
    PalmDetect palmDetect;
    LandmarkDetect landmarkDetect;

    Image[] frames;

    void Start()
    {
        string palmPath = Path.Combine(Application.streamingAssetsPath, palmModelFile);
        palmDetect = new PalmDetect(palmPath, anchorCsv.text);

        string landmarkPath = Path.Combine(Application.streamingAssetsPath, landmarkModelFile);
        landmarkDetect = new LandmarkDetect(landmarkPath);

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
        landmarkDetect?.Dispose();
    }

    void Update()
    {
        var resizeOptions = palmDetect.ResizeOptions;
        resizeOptions.rotationDegree = webcamTexture.videoRotationAngle;
        palmDetect.ResizeOptions = resizeOptions;

        palmDetect.Invoke(webcamTexture);

        var palms = palmDetect.GetResults(0.7f, 0.3f);
        UpdateFrame(palms);

        cameraView.material = palmDetect.transformMat;

        if (palms.Count <= 0)
        {
            return;
        }
        landmarkDetect.Invoke(palmDetect.Input0);
    }

    void UpdateFrame(List<PalmDetect.Palm> palms)
    {
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

        var kpOffset = -rt.anchoredPosition + new Vector2(-rt.sizeDelta.x, rt.sizeDelta.y) * 0.5f;
        for (int i = 0; i < 7; i++)
        {
            var child = (RectTransform)rt.GetChild(i);
            var kp = palm.keypoints[i];
            kp.y = 1.0f - kp.y; // invert Y
            child.anchoredPosition = (kp * size - size * 0.5f) + kpOffset;
        }
    }

}
