# TensorFlow Lite for Unity Samples

![mnist gif](https://imgur.com/yi2MtCF.gif)

---

![ssd gif](https://imgur.com/8m2oqwo.gif)

Tested on macOS/iOS  
Unity 2910.2.8

## How to build tensorflow lite for Unity

Pre-build library is included. see following instructions if you want build your own lib.

### macOS

```sh
# Core Lib
bazel build -c opt --cxxopt=--std=c++11 tensorflow/lite/experimental/c:libtensorflowlite_c.so

# Use this branch to build metal GPU delegate dynamic library
# https://github.com/asus4/tensorflow/tree/tflite-macos-metal-delegate
bazel 'build' -c opt --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always --cxxopt=-std=c++14 --apple_platform_type=macos '//tensorflow/lite/delegates/gpu:tensorflow_lite_gpu_dylib'
```

then rename libtensorflowlite_c.so to libtensorflowlite_c.bundle

### iOS

Download pre-build framework from CocoaPods

```ruby
# Sample Podfile

platform :ios, '10.0'

target 'TfLiteSample' do
    pod 'TensorFlowLiteObjC', '0.0.1-nightly'
end
```

### Android

If you do not have the Android SDK and NDK, intall Android Studio, SDK and NDK.

```sh
# Configure the Android SDK path by running configure script at repository root
./configure

# Build experimental
bazel build -c opt --cxxopt=--std=c++11 --config=android_arm //tensorflow/lite/experimental/c:libtensorflowlite_c.so

# Build GPU delegate
bazel build -c opt --cxxopt=--std=c++11 --config=android_arm64 //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so
```

## License

Samples folder `Assets/Samples/*` is licensed under MIT

Other licenses

- [TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)
- [*.tflite Models](https://www.tensorflow.org/lite/models)
