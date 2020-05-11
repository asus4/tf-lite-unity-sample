#!/usr/bin/env python3


import argparse
import os
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

    run_cmd('bazel build -c opt --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always --cxxopt=-std=c++14 --apple_platform_type=macos \'//tensorflow/lite/delegates/gpu:tensorflow_lite_gpu_dylib\'')
    copy('bazel-bin/tensorflow/lite/delegates/gpu/tensorflow_lite_gpu_dylib.dylib', 'macOS/libtensorflowlite_metal_delegate.dylib')

def build_ios():
    # run_cmd('bazel build -c opt --cxxopt=--std=c++11 tensorflow/lite/c:tensorflowlite_c')
    # copy('bazel-bin/tensorflow/lite/c/libtensorflowlite_c.dylib', 'macOS/libtensorflowlite_c.dylib')

    run_cmd('bazel build -c opt --cpu ios_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --copt=-fembed-bitcode --linkopt -s --strip always --cxxopt=-std=c++14 //tensorflow/lite/delegates/gpu:tensorflow_lite_gpu_framework --apple_platform_type=ios')
    unzip('bazel-bin/tensorflow/lite/delegates/gpu/tensorflow_lite_gpu_framework.zip', 'ios')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build TensorFlow libraries for Unity')
    parser.add_argument('--tfpath',default='../tensorflow')
    parser.add_argument('-macos', action="store_true", default=False)
    parser.add_argument('-ios', action="store_true", default=False)

    args = parser.parse_args()
    TENSORFLOW_PATH = os.path.abspath(args.tfpath) 

    if args.macos:
        print('Build macOS')
        build_mac()
    
    if args.ios:
        print('Build ios')
        build_ios()
    
