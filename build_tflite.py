#!/usr/bin/env python3

import argparse
import os
import platform
import shlex
import subprocess

PLUGIN_PATH=f'{os.getcwd()}/Assets/TensorFlowLite/Plugins'
TENSORFLOW_PATH=''

def run_cmd(cmd):
    args = shlex.split(cmd)
    subprocess.call(args, cwd=TENSORFLOW_PATH)

def copy(from_tf, to_unity):
    subprocess.call(['cp', '-vf', f'{TENSORFLOW_PATH}/{from_tf}', f'{PLUGIN_PATH}/{to_unity}'])

def unzip(from_tf, to_unity):
    subprocess.call(['unzip', '-o', f'{TENSORFLOW_PATH}/{from_tf}', '-d' f'{PLUGIN_PATH}/{to_unity}'])

def build_mac():
    run_cmd('bazel build -c opt --cxxopt=--std=c++11 tensorflow/lite/c:tensorflowlite_c')
    copy('bazel-bin/tensorflow/lite/c/libtensorflowlite_c.dylib', 'macOS/libtensorflowlite_c.dylib')

    run_cmd('bazel build -c opt --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always --cxxopt=-std=c++14 --apple_platform_type=macos //tensorflow/lite/delegates/gpu:tensorflow_lite_gpu_dylib')
    copy('bazel-bin/tensorflow/lite/delegates/gpu/tensorflow_lite_gpu_dylib.dylib', 'macOS/libtensorflowlite_metal_delegate.dylib')

def build_windows():
    run_cmd('bazel build -c opt --cxxopt=--std=c++11 tensorflow/lite/c:tensorflowlite_c')
    copy('bazel-bin/tensorflow/lite/c/tensorflowlite_c.dll', 'Windows/libtensorflowlite_c.dll')


def build_ios():
    run_cmd('bazel build --config=ios_fat -c opt //tensorflow/lite/experimental/ios:TensorFlowLiteC_framework')
    unzip('bazel-bin/tensorflow/lite/experimental/ios/TensorFlowLiteC_framework.zip', 'iOS')

    run_cmd('bazel build -c opt --config=ios_fat --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --copt=-fembed-bitcode --linkopt -s --strip always --cxxopt=-std=c++14 //tensorflow/lite/delegates/gpu:tensorflow_lite_gpu_framework --apple_platform_type=ios')
    unzip('bazel-bin/tensorflow/lite/delegates/gpu/tensorflow_lite_gpu_framework.zip', 'iOS')

def build_android():
    run_cmd('bazel build -c opt --cxxopt=--std=c++11 --config=android_arm64 //tensorflow/lite/c:libtensorflowlite_c.so')
    copy('bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so', 'Android')
    
    run_cmd('bazel build -c opt --config android_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so')
    copy('bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so', 'Android')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Update TensorFlow Lite libraries for Unity')
    parser.add_argument('--tfpath', default = '../tensorflow', type = str,
                        help = 'The path of the TensorFlow repository')
    parser.add_argument('-macos', action = "store_true", default = False,
                        help = 'Build macOS')
    parser.add_argument('-windows', action = "store_true", default = False,
                        help = 'Build Windows')
    parser.add_argument('-ios', action = "store_true", default = False,
                        help = 'Build iOS')
    parser.add_argument('-android', action = "store_true", default = False,
                        help = 'Build Android')

    args = parser.parse_args()
    TENSORFLOW_PATH = os.path.abspath(args.tfpath) 

    platform_name = platform.system()

    if args.macos:
        assert platform_name is 'Darwin', f'-macos not suppoted on the platfrom: {platform_name}'
        print('Build macOS')
        build_mac()
    
    if args.windows:
        assert platform_name is 'Windows', f'-windows not suppoted on the platfrom: {platform_name}'
        print('Build Windows')
        build_windows()
    
    if args.ios:
        assert platform_name is 'Darwin', f'-ios not suppoted on the platfrom: {platform_name}'
        # Need to set iOS build option in ./configure
        print('Build iOS')
        build_ios()
    
    if args.android:
        # Need to set Android build option in ./configure
        print('Build Android')
        build_android()
