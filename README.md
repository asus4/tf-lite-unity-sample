# TensorFlow Lite for Unity Samples

![mnist gif](https://imgur.com/yi2MtCF.gif)

---

![ssd gif](https://imgur.com/8m2oqwo.gif)

Tested on macOS/iOS  
Unity 2910.2.8

## How to build tensorflow lite for Unity

### macOS

```sh
bazel build -c opt --cxxopt=--std=c++11 tensorflow/lite/experimental/c:libtensorflowlite_c.so
```

then rename libtensorflowlite_c.so to libtensorflowlite_c.bundle

### iOS

Download pre-build framework from CocoaPods

```Podfile
# Sample Podfile

platform :ios, '10.0'

target 'TfLiteSample' do
    pod 'TensorFlowLiteObjC', '0.0.1-nightly'
end
```

## License

Samples folder `Assets/Samples/*` is licensed under MIT

Other licenses

- [TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)
- [*.tflite Models](https://www.tensorflow.org/lite/models)
