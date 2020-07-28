# TensorFlow Lite for Unity Samples

[![openupm](https://img.shields.io/npm/v/com.github.asus4.tflite?label=openupm&registry_uri=https://package.openupm.com)](https://openupm.com/packages/com.github.asus4.tflite/)

Porting of ["TensorFlow Lite Examples"](https://www.tensorflow.org/lite/examples) to Unity. and some utilities for Unity.

Tested on  

- macOS / iOS / Android / Ubuntu / Windows  
- Unity 2019.3.3f1  
- TensorFlow 2.2.0  

## Install TensorFlow Lite for Unity

- Clone or Download this repository with examples
  - Need [Git-LFS](https://git-lfs.github.com/) to build for iOS
- Or core library is available on [OpenUPM](https://openupm.com/packages/com.github.asus4.tflite/)

## Update TensorFlow Lite

Pre-built libraries are included. If you want to use the latest TFLite,

1. Clone [TensorFlow library](https://github.com/tensorflow/tensorflow/)
2. Run `./configure` in the TensorFlow library
3. Run `./build_tflite.py` (Python3) to build for each platform

  ```sh
  # Update iOS, Andoid and macOS
  ./build_tflte.py --tfpath ../tensorflow -ios -android -macos
  ```

- To build macOS Metal Delegate on TensorFlow v2.3.0 or later, See [the issue](https://github.com/tensorflow/tensorflow/issues/41039)

## Misc

\[Android\] You can see logs from tflite by filtering with "tflite"  

```bash
adb logcat Unity:V tflite:V "*:S"
```

## Show Cases

__MNIST__  
![Mnist](https://imgur.com/yi2MtCF.gif)

__SSD Object Detection__  
![SSD](https://imgur.com/Omeatqc.gif)

__DeepLab Semantic Segmentation__  
![DeepLab](https://imgur.com/tH1Z8NG.gif)

__Style Transfer__  
![styletransfter](https://i.imgur.com/SOLMjZi.gif)

__Hand Tracking__  
![handtracking](https://i.imgur.com/KDOCx2U.gif)

## License

Samples folder `Assets/Samples/*` is licensed under MIT

Other licenses

- [TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)
- [Assets/StreamingAssets/*.tflite Models](https://www.tensorflow.org/lite/models)
- [Assets/StreamingAssets/mediapipe Models](https://github.com/google/mediapipe)
  - Some MediaPipe C# codes are based on [terryky/tflite_gles_app](https://github.com/terryky/tflite_gles_app)
