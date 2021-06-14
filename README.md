# TensorFlow Lite for Unity Samples

[![openupm](https://img.shields.io/npm/v/com.github.asus4.tflite?label=openupm&registry_uri=https://package.openupm.com)](https://openupm.com/packages/com.github.asus4.tflite/)

Porting of ["TensorFlow Lite Examples"](https://www.tensorflow.org/lite/examples) to Unity. and some utilities for Unity.

Tested on  

- iOS / Android / macOS / Windows  
- Unity 2019.4.24f1
- TensorFlow 2.4.0

Samples

- TensorFlow
  - MNIST
  - SSD Object Detection
  - DeepLab
  - PoseNet
  - Style Transfer
  - Text Classification
  - Bert Question and Answer
  - Super Resolution
- MediaPipe
  - Hand Tracking
  - Blaze Face
  - Face Mesh
  - Blaze Pose (Upper body)
- MLKit
  - Blaze Pose (Full body)
- Meet Segmentation

Included prebuilt libraries

| | iOS | Android | macOS | Ubuntu | Windows |
|---|:---:|:---:|:---:|:---:|:---:|
| Core CPU |✅|✅|✅|✅|✅|
| Metal Delegate |✅| - |✅| - | - |
| OpenGL Delegate | - |✅| - | - | - |
| NNAPI Delegate | - |✅| - | - | - |

- All libraries except iOS are targeted 64bit platform: arm64 or x86_64.

## Install TensorFlow Lite for Unity

- Clone this repository with examples
  - Requires installing [Git-LFS](https://git-lfs.github.com/)
- The TFLite core library is available on:
  - [OpenUPM](https://openupm.com/packages/com.github.asus4.tflite/)  
  Run `openupm add com.github.asus4.tflite` from the command line.
  - Or add git URL from the Package Maneger UI: `https://github.com/asus4/tf-lite-unity-sample.git?path=/Packages/com.github.asus4.tflite`

## Build TensorFlow Lite libraries

Pre-built libraries are included. If you want to build the latest TFLite,

1. Clone [TensorFlow library](https://github.com/tensorflow/tensorflow/)
2. Run `./configure` in the TensorFlow library
3. Run `./build_tflite.py` (Python3) to build for each platform

  ```sh
  # Update iOS, Andoid and macOS
  ./build_tflte.py --tfpath ../tensorflow -ios -android -macos

  # Build with XNNPACK
  ./build_tflte.py --tfpath ../tensorflow -macos -xnnpack
  ```

- To build macOS Metal Delegate on TensorFlow v2.3.0 or later, You need to apply following changes [the issue](https://github.com/tensorflow/tensorflow/issues/41039#issuecomment-664701908)

## TIPS

\[Android\] You can see logs from tflite by filtering with "tflite"  

```bash
# Filtering logcat only Unity and tflite
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

```markdown
MIT License

Copyright (c) 2021 Koki Ibukuro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Other Licenses

- [TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/LICENSE): Apache License 2.0
- [MediaPipe](https://github.com/google/mediapipe/blob/master/LICENSE): Apache License 2.0
  - Some MediaPipe C# codes are based on [terryky/tflite_gles_app](https://github.com/terryky/tflite_gles_app)

Model Licenses

📌 : Each TensorFlow Lite model might have a different license. Please check the license of the model you use.

- [Official TFlite Models](https://www.tensorflow.org/lite/examples)
  - Bert
  - SSD
  - DeepLab Lab
  - MNIST
  - Style Transfer
  - PoseNet
  - Text classification
  - Smart Reply
- [MediaPipe Models](https://github.com/google/mediapipe)
  - Blaze Pose
  - Face Mesh
  - Hand Tracking
- [Meet Segmentation Model](https://drive.google.com/file/d/1lnP1bRi9CSqQQXUHa13159vLELYDgDu0/view)
  - Using the modified model from [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) to remove the custom post-process.
