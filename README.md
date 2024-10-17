# TensorFlow Lite for Unity Samples

[![npm](https://img.shields.io/npm/v/com.github.asus4.tflite?label=npm)](https://www.npmjs.com/package/com.github.asus4.tflite)

Porting of ["TensorFlow Lite Examples"](https://www.tensorflow.org/lite/examples) and some utilities for Unity.

Tested on  

- iOS / Android / macOS / Windows / Linux
- Unity 2022.3.22f1
- TensorFlow 2.16.1

Included examples:

- TensorFlow
  - MNIST
  - EfficientDet Object Detection
  - DeepLab
  - MoveNet
  - Style Transfer
  - Text Classification
  - Bert Question and Answer
  - Super Resolution
  - Audio Classification
- MediaPipe
  - Hand Tracking
  - Blaze Face
  - Face Mesh
  - Blaze Pose (Full body)
  - Selfie Segmentation

Included prebuilt libraries:

| | iOS | Android | macOS | Ubuntu | Windows |
|---|:---:|:---:|:---:|:---:|:---:|
| Core CPU |✅|✅|✅|✅|✅|
| Metal Delegate |✅| - |✅| - | - |
| GPU Delegate | - |✅| - | ✅ Experimental | - |
| NNAPI Delegate | - |✅| - | - | - |

- You need to install OpenGL ES and OpenCL to run GPU Delegate on Linux. See [MediaPipe](https://google.github.io/mediapipe/getting_started/gpu_support.html#opengl-es-setup-on-linux-desktop) for details.

## Install TensorFlow Lite for Unity

> [!IMPORTANT]  
> You need to install [Git-LFS](https://git-lfs.github.com/).

- To try all examples, clone this repository with [Git-LFS](https://git-lfs.github.com/).
- If you need TensorFlow Lite libraries via UPM, open the file `Packages/manifest.json` and add the following lines into the `scopedRegistries` and `dependencies` sections.

```json
{
  "scopedRegistries": [
    {
      "name": "package.openupm.com",
      "url": "https://package.openupm.com",
      "scopes": [
        "com.cysharp.unitask"
      ]
    },
    {
      "name": "npm",
      "url": "https://registry.npmjs.com",
      "scopes": [
        "com.github.asus4"
      ]
    }
  ],
  "dependencies": {
    // Core TensorFlow Lite libraries
    "com.github.asus4.tflite": "2.17.0",
    // Optional: Utilities for TFLite
    "com.github.asus4.tflite.common": "2.17.0",
    // Optional: Utilities for MediaPipe
    "com.github.asus4.mediapipe": "2.17.0",
    // Optional: Async methods are available only when UniTask is installed
    "com.cysharp.unitask": "2.5.10",
    ...// other dependencies
  }
}
```

## Build TensorFlow Lite libraries yourself

Pre-built libraries are included in the UPM package. Also, you can find TFLite libraries at [tflite-runtime-builder](https://github.com/DeNA/tflite-runtime-builder/releases) from TFLite v2.14.0 or later.

If you want to build the latest TFLite yourself, Follow the below instructions:

1. Clone [TensorFlow library](https://github.com/tensorflow/tensorflow/)
2. Run `./configure` in the TensorFlow library
3. Run `./build_tflite.py` (Python3) to build for each platform

  ```sh
  # Update iOS, Android and macOS
  ./build_tflite.py --tfpath ../tensorflow -ios -android -macos

  # Build with XNNPACK
  ./build_tflite.py --tfpath ../tensorflow -macos -xnnpack
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

Copyright (c) 2024 Koki Ibukuro

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
  - Selfie Segmentation
    - Using the modified model from [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) to remove the custom post-process.
