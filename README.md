# TensorFlow Lite for Unity Samples

Porting of ["TensorFlow Lite Examples"](https://www.tensorflow.org/lite/examples) to Unity. and some utilities for Unity.

Tested on 
- macOS / iOS / Android / Ubuntu / Windows  
- Unity 2019.3.3f1  
- TensorFlow 2.2.0  

Please install [Git-LFS](https://git-lfs.github.com/) to build for iOS 

## How to build tensorflow lite for Unity

Pre-built libraries are included. If you want to use the latest TFLite,

1. Clone tensorflow library
2. Run `./configure` in tensorflow and
3. Run `./build_tflite.py` (Python3) to build for each platform

  ```sh
  # Update iOS, Andoid and macOS
  ./build_tflte.py --tfpath ../tensorflow -ios -android -macos
  ```

## Show Case

__MNIST__  
![Mnist](https://imgur.com/yi2MtCF.gif)

__SSD Object Detection__  
![SSD](https://imgur.com/Omeatqc.gif)

__DeepLab Semantic Segmentation__  
![DeepLab](https://imgur.com/tH1Z8NG.gif)

__Style Transfer__  
![styletransfter](https://i.imgur.com/SOLMjZi.gif)

## License

Samples folder `Assets/Samples/*` is licensed under MIT

Other licenses

- [TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)
- [*.tflite Models](https://www.tensorflow.org/lite/models)

