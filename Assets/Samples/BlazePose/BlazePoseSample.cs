using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;

/// <summary>
/// BlazePose form MediaPipe
/// https://github.com/google/mediapipe
/// </summary>
public class BlazePoseSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string poseDetectionModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField, FilePopup("*.tflite")] string poseLandmarkModelFile = "coco_ssd_mobilenet_quant.tflite";

    [SerializeField] RawImage cameraView = null;
    [SerializeField] Image framePrefab = null;
    [SerializeField] RawImage debugPalmView = null;
    [SerializeField] Mesh jointMesh = null;
    [SerializeField] Material jointMaterial = null;

    WebCamTexture webcamTexture;
    PoseDetect poseDetect;
    PoseLandmark poseLandmark;

    void Start()
    {
        string detectionPath = Path.Combine(Application.streamingAssetsPath, poseDetectionModelFile);
        poseDetect = new PoseDetect(detectionPath);
        string landmarkPath = Path.Combine(Application.streamingAssetsPath, poseLandmarkModelFile);
        poseLandmark = new PoseLandmark(landmarkPath);

        string cameraName = WebCamUtil.FindName(new WebCamUtil.PreferSpec()
        {
            isFrontFacing = false,
            kind = WebCamKind.WideAngle,
        });
        webcamTexture = new WebCamTexture(cameraName, 1280, 720, 30);
        cameraView.texture = webcamTexture;
        webcamTexture.Play();
        Debug.Log($"Starting camera: {cameraName}");
    }

    void OnDestroy()
    {
        webcamTexture?.Stop();
        poseDetect?.Dispose();
        poseLandmark?.Dispose();
    }

    void Update()
    {
        var resizeOptions = poseDetect.ResizeOptions;
        resizeOptions.rotationDegree = webcamTexture.videoRotationAngle;
        poseDetect.ResizeOptions = resizeOptions;

        poseDetect.Invoke(webcamTexture);
        cameraView.material = poseDetect.transformMat;


    }
}
