# TensorFlow Lite for Unity Samples

[![openupm](https://img.shields.io/npm/v/com.github.asus4.tflite?label=openupm&registry_uri=https://package.openupm.com)](https://openupm.com/packages/com.github.asus4.tflite/)

Porting of ["TensorFlow Lite Examples"](https://www.tensorflow.org/lite/examples) to Unity. and some utilities for Unity.

Tested on  

- iOS / Android / macOS / Ubuntu / Windows  
- Unity 2019.4.5f1  
- TensorFlow 2.3.0

Included prebuilt libraries

| | iOS | Android | macOS | Ubuntu | Windows |
|---|:---:|:---:|:---:|:---:|:---:|
| Core CPU |✅|✅|✅|✅|✅|
| Metal Delegate |✅| - |✅| - | - |
| OpenGL Delegate | - |✅| - | - | - |
| NNAPI Delegate | - |✅| - | - | - |

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

- To build macOS Metal Delegate on TensorFlow v2.3.0 or later, You need to apply following changes [the issue](https://github.com/tensorflow/tensorflow/issues/41039#issuecomment-664701908)

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
![handtracking](https://user-images.githubusercontent.com/357497/89078175-28179780-d384-11ea-8a35-8b48a31aa52d.gif)

__BERT__  
![BERT](https://user-images.githubusercontent.com/357497/89077837-6496c380-d383-11ea-96f8-a5ae6e61d603.png)

## License

Samples folder `Assets/Samples/*` is licensed under MIT

Other licenses

- [TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)
- [Assets/StreamingAssets/*.tflite Models](https://www.tensorflow.org/lite/models)
- [Assets/StreamingAssets/mediapipe Models](https://github.com/google/mediapipe)
  - Some MediaPipe C# codes are based on [terryky/tflite_gles_app](https://github.com/terryky/tflite_gles_app)
