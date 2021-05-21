#!/usr/bin/env python3

import argparse
import os
import platform
import shlex
import subprocess

PLUGIN_PATH=f'{os.getcwd()}/Packages/com.github.asus4.tflite/Plugins'
TENSORFLOW_PATH=''

def run_cmd(cmd):
    print(cmd)
    args = shlex.split(cmd)
    subprocess.call(args, cwd=TENSORFLOW_PATH)

def copy(from_tf, to_unity):
    subprocess.call(['cp', '-vf', f'{TENSORFLOW_PATH}/{from_tf}', f'{PLUGIN_PATH}/{to_unity}'])

def unzip(from_tf, to_unity):
    subprocess.call(['unzip', '-o', f'{TENSORFLOW_PATH}/{from_tf}', '-d' f'{PLUGIN_PATH}/{to_unity}'])

def patch(file_path, target_str, patched_str):
    with open(file_path, 'r', encoding='utf-8') as file:
        source = file.read()
    source = source.replace(target_str, patched_str)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(source)

def build_mac(enable_xnnpack = False):
    # Main
    option_xnnpack = 'true' if enable_xnnpack else 'false'
    run_cmd(f'bazel build --config=macos -c opt --define tflite_with_xnnpack={option_xnnpack} tensorflow/lite/c:tensorflowlite_c')
    copy('bazel-bin/tensorflow/lite/c/libtensorflowlite_c.dylib', 'macOS/libtensorflowlite_c.dylib')

    # Metal Delegate
    # v2.3.0 or later, Need to apply the following patch to build metal delegate
    # For further info
    # https://github.com/tensorflow/tensorflow/issues/41039#issuecomment-664701908
    cpuinfo_file = f'{TENSORFLOW_PATH}/third_party/cpuinfo/BUILD.bazel'
    original = '"cpu": "darwin",'
    patched = '"cpu": "darwin_x86_64",'
    patch(cpuinfo_file, original, patched)
    # Build Metal Delegate
    run_cmd('bazel build --config=macos -c opt --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=default --linkopt -s --strip always --apple_platform_type=macos //tensorflow/lite/delegates/gpu:tensorflow_lite_gpu_dylib')
    copy('bazel-bin/tensorflow/lite/delegates/gpu/tensorflow_lite_gpu_dylib.dylib', 'macOS/libtensorflowlite_metal_delegate.dylib')
    # Restore it
    patch(cpuinfo_file, patched, original)

def build_windows(enable_xnnpack = False):
    # Main
    option_xnnpack = 'true' if enable_xnnpack else 'false'
    run_cmd(f'bazel build -c opt --define tflite_with_xnnpack={option_xnnpack} tensorflow/lite/c:tensorflowlite_c')
    copy('bazel-bin/tensorflow/lite/c/tensorflowlite_c.dll', 'Windows/libtensorflowlite_c.dll')
    # TODO GPU Delegate

def build_linux():
    # Testd on Ubuntu 18.04.5 LTS
    # Main
    run_cmd('bazel build -c opt --cxxopt=--std=c++11 tensorflow/lite/c:tensorflowlite_c')
    copy('bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so', 'Linux/libtensorflowlite_c.so')
    # TODO GPU Delegate

def build_ios():
    # Main
    run_cmd('bazel build --config=ios_fat -c opt //tensorflow/lite/ios:TensorFlowLiteC_framework')
    unzip('bazel-bin/tensorflow/lite/ios/TensorFlowLiteC_framework.zip', 'iOS')
    # Metal Delegate
    run_cmd('bazel build -c opt --config=ios_fat //tensorflow/lite/ios:TensorFlowLiteCMetal_framework')
    unzip('bazel-bin/tensorflow/lite/ios/TensorFlowLiteCMetal_framework.zip', 'iOS')
    # CoreML Delegate
    # run_cmd('bazel build -c opt --config=ios_fat //tensorflow/lite/ios:TensorFlowLiteCCoreML_framework')
    # unzip('bazel-bin/tensorflow/lite/ios/TensorFlowLiteCCoreML_framework.zip', 'iOS')
    # SelectOps Delegate
    # run_cmd('bazel build -c opt --config=ios --ios_multi_cpus=armv7,arm64,x86_64 //tensorflow/lite/ios:TensorFlowLiteSelectTfOps_framework')
    # unzip('bazel-bin/tensorflow/lite/ios/TensorFlowLiteSelectTfOps_framework.zip', 'iOS')

def build_android(enable_xnnpack = False):
    # Main
    option_xnnpack = 'true' if enable_xnnpack else 'false'
    run_cmd(f'bazel build -c opt --config=android_arm64 --define tflite_with_xnnpack={option_xnnpack} //tensorflow/lite/c:libtensorflowlite_c.so')
    copy('bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so', 'Android')
    # GPU Delegate
    run_cmd('bazel build -c opt --config=android_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so')
    copy('bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so', 'Android')
    # GL Delegate
    run_cmd('bazel build -c opt --config=android_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_gl.so')
    copy('bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_gl.so', 'Android')
    
    # NNAPI Delegate
    # run_cmd('bazel build -c opt --config=android_arm64 //tensorflow/lite/delegates/nnapi:nnapi_delegate')
    # copy('bazel-bin/tensorflow/lite/delegates/nnapi/libnnapi_delegate.so', 'Android')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Update TensorFlow Lite libraries for Unity')
    parser.add_argument('--tfpath', default = '../tensorflow', type = str,
                        help = 'The path of the TensorFlow repository')
    parser.add_argument('-macos', action = "store_true", default = False,
                        help = 'Build macOS')
    parser.add_argument('-windows', action = "store_true", default = False,
                        help = 'Build Windows')
    parser.add_argument('-linux', action = "store_true", default = False,
                        help = 'Build Linux')
    parser.add_argument('-ios', action = "store_true", default = False,
                        help = 'Build iOS')
    parser.add_argument('-android', action = "store_true", default = False,
                        help = 'Build Android')
    parser.add_argument('-xnnpack', action = "store_true", default = False,
                        help = 'Build with XNNPACK')

    args = parser.parse_args()
    TENSORFLOW_PATH = os.path.abspath(args.tfpath) 

    platform_name = platform.system()

    if args.macos:
        assert platform_name == 'Darwin', f'-macos not suppoted on the platfrom: {platform_name}'
        print('Build macOS')
        build_mac(args.xnnpack)
    
    if args.windows:
        assert platform_name == 'Windows', f'-windows not suppoted on the platfrom: {platform_name}'
        print('Build Windows')
        build_windows(args.xnnpack)
    
    if args.linux:
        assert platform_name == 'Linux', f'-linux not suppoted on the platfrom: {platform_name}'
        print('Build Linux')
        build_linux()
    
    if args.ios:
        assert platform_name == 'Darwin', f'-ios not suppoted on the platfrom: {platform_name}'
        # Need to set iOS build option in ./configure
        print('Build iOS')
        build_ios()
    
    if args.android:
        # Need to set Android build option in ./configure
        print('Build Android')
        build_android(args.xnnpack)
