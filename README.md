# TensorFlow Lite for Unity Samples

![mnist gif](https://imgur.com/yi2MtCF.gif)

---

![ssd gif](https://imgur.com/8m2oqwo.gif)

Tested on macOS / iOS / Android / Ubuntu / Windows  
Unity 2019.3.3f1  
TensorFlow 2.2.0  

## How to build tensorflow lite for Unity

The pre-built libraries are included. if you want to use the latest TFLite,

1. Clone tensorflow library
2. Run `./configure` in tensorflow and
3. Run `./build_tflite.py` (Python3) to build for each platform

  ```sh
  # Update iOS, Andoid and macOS
  ./build_tflte.py --tfpath ../tensorflow -ios -android -macos
  ```

## License

Samples folder `Assets/Samples/*` is licensed under MIT

Other licenses

- [TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)
- [*.tflite Models](https://www.tensorflow.org/lite/models)
