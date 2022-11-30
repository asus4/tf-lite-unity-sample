#!/usr/bin/env python3

import argparse
import glob
import os
import platform
import shlex
import shutil
import subprocess
import sys

PLUGIN_PATH=f'{os.getcwd()}/Packages/com.github.asus4.tflite/Plugins'
TENSORFLOW_PATH=''

def run_cmd(cmd):
    print(cmd)
    args = shlex.split(cmd)
    is_shell = platform.system() == 'Windows'
    ret = subprocess.call(args, cwd=TENSORFLOW_PATH, shell=is_shell)
    if ret != 0:
        sys.exit(ret)

def copy(from_tf, to_unity):
    shutil.copy(f'{TENSORFLOW_PATH}/{from_tf}', f'{PLUGIN_PATH}/{to_unity}')

def unzip(from_tf, to_unity):
    subprocess.call(['unzip', '-o', f'{TENSORFLOW_PATH}/{from_tf}', '-d' f'{PLUGIN_PATH}/{to_unity}'])

def patch(file_path, target_str, patched_str):
    with open(file_path, 'r', encoding='utf-8') as file:
        source = file.read()
    source = source.replace(target_str, patched_str)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(source)

def build_mac(enable_xnnpack = True):
    # Main
    option_xnnpack = 'true' if enable_xnnpack else 'false'
    run_cmd(f'bazel build --config=macos --cpu=darwin -c opt --define tflite_with_xnnpack={option_xnnpack} tensorflow/lite/c:tensorflowlite_c')
    run_cmd(f'bazel build --config=macos_arm64 --cpu=darwin_arm64 -c opt --define tflite_with_xnnpack={option_xnnpack} tensorflow/lite/c:tensorflowlite_c')
    dylib_bin_path = 'bin/tensorflow/lite/c/libtensorflowlite_c.dylib'
    run_cmd(f'lipo -create -output {PLUGIN_PATH}/macOS/libtensorflowlite_c.dylib bazel-out/darwin-opt/{dylib_bin_path} bazel-out/darwin_arm64-opt/{dylib_bin_path}')

    # Metal Delegate
    # v2.3.0 or later, Need to apply the following patch to build metal delegate
    # For further info
    # https://github.com/tensorflow/tensorflow/issues/41039#issuecomment-664701908
    cpuinfo_file = f'{TENSORFLOW_PATH}/bazel-tensorflow/external/cpuinfo/BUILD.bazel'
    original = '"cpu": "darwin",'
    patched = '"cpu": "darwin_x86_64",'
    patch(cpuinfo_file, original, patched)
    run_cmd('bazel build --config=macos --cpu=darwin_x86_64 --macos_cpus=x86_64 --apple_platform_type=macos --cxxopt=--std=c++17 -c opt --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=default --linkopt -s --strip always //tensorflow/lite/delegates/gpu:tensorflow_lite_gpu_dylib')
    run_cmd('bazel build --config=macos_arm64 --cpu=darwin_arm64  --macos_cpus=arm64 --apple_platform_type=macos --cxxopt=--std=c++17 -c opt --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=default --linkopt -s --strip always //tensorflow/lite/delegates/gpu:tensorflow_lite_gpu_dylib')
    # Export path contains postfix like `applebin_macos-darwin_arm64-opt-ST-*`
    metal_delegate_pathes = glob.glob(f'{TENSORFLOW_PATH}/bazel-out/applebin_macos-darwin*/bin/tensorflow/lite/delegates/gpu/tensorflow_lite_gpu_dylib.dylib')
    print(metal_delegate_pathes)
    run_cmd(f'lipo -create -output {PLUGIN_PATH}/macOS/libtensorflowlite_metal_delegate.dylib ' + ' '.join(metal_delegate_pathes))
    # Restore patch
    patch(cpuinfo_file, patched, original)

def build_windows(enable_xnnpack = True):
    # Main
    option_xnnpack = 'true' if enable_xnnpack else 'false'
    run_cmd(f'bazel build -c opt --define tflite_with_xnnpack={option_xnnpack} tensorflow/lite/c:tensorflowlite_c')
    copy('bazel-bin/tensorflow/lite/c/tensorflowlite_c.dll', 'Windows/libtensorflowlite_c.dll')
    # TODO support GPU Delegate

def build_linux(enable_xnnpack = True):
    # Tested on Ubuntu 18.04.5 LTS
    # Main
    option_xnnpack = 'true' if enable_xnnpack else 'false'
    run_cmd(f'bazel build -c opt --define tflite_with_xnnpack={option_xnnpack} tensorflow/lite/c:tensorflowlite_c')
    copy('bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so', 'Linux/x86_64/libtensorflowlite_c.so')

    # For Embedded Linux
    run_cmd(f'bazel build --config=elinux_aarch64 -c opt --define tflite_with_xnnpack={option_xnnpack} tensorflow/lite/c:tensorflowlite_c')
    copy('bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so', 'Linux/arm64/libtensorflowlite_c.so')

    # GPU Delegate
    # See MediaPipe docs to setup EGL on Linux https://google.github.io/mediapipe/getting_started/gpu_support.html#opengl-es-setup-on-linux-desktop
    # Also, you need link EGL and GLESv2 for Linux platform, will make a patch for this
    # https://github.com/tensorflow/tensorflow/blob/5850c0ba26745f92456234c34ed258b472f07487/tensorflow/lite/delegates/gpu/build_defs.bzl#L3-L15
    run_cmd('bazel build --config=linux -c opt --copt -Os --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 --copt -DCL_TARGET_OPENCL_VERSION=210 --copt -fvisibility=default --linkopt -s --strip always //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so')
    copy('bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so', 'Linux/x86_64/libtensorflowlite_gpu_delegate.so')
    
    # GPU Delegate for Embedded Linux, the cross-compile will not work as EGL it not linked correctly with aarch64_linux_toolchain
    # run_cmd('bazel build --config=elinux_aarch64 -c opt --copt -Os --copt -DCL_DELEGATE_NO_GL --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 --copt -DCL_TARGET_OPENCL_VERSION=210 --copt -fvisibility=default --linkopt -s --strip always //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_gl.so')


def build_ios():
    # Main
    run_cmd('bazel build -c opt --config=ios_arm64 --cxxopt=--std=c++17 //tensorflow/lite/ios:TensorFlowLiteC_framework')
    unzip('bazel-bin/tensorflow/lite/ios/TensorFlowLiteC_framework.zip', 'iOS')
    # Metal Delegate
    run_cmd('bazel build -c opt --config=ios_arm64 --cxxopt=--std=c++17 //tensorflow/lite/ios:TensorFlowLiteCMetal_framework')
    unzip('bazel-bin/tensorflow/lite/ios/TensorFlowLiteCMetal_framework.zip', 'iOS')
    # CoreML Delegate
    run_cmd('bazel build -c opt --config=ios_arm64 --cxxopt=--std=c++17 //tensorflow/lite/ios:TensorFlowLiteCCoreML_framework')
    unzip('bazel-bin/tensorflow/lite/ios/TensorFlowLiteCCoreML_framework.zip', 'iOS')
    # SelectOps Delegate
    # run_cmd('bazel build -c opt --config=ios --ios_multi_cpus=armv7,arm64,x86_64 //tensorflow/lite/ios:TensorFlowLiteSelectTfOps_framework')
    # unzip('bazel-bin/tensorflow/lite/ios/TensorFlowLiteSelectTfOps_framework.zip', 'iOS')

def build_android(enable_xnnpack = True):
    # Main
    option_xnnpack = 'true' if enable_xnnpack else 'false'
    run_cmd(f'bazel build -c opt --fat_apk_cpu=arm64-v8a,armeabi-v7a,x86_64 //tensorflow/lite/java:tensorflow-lite')
    copy('bazel-bin/tensorflow/lite/java/tensorflow-lite.aar', 'Android')

    # GPU Delegate
    run_cmd('bazel build -c opt --fat_apk_cpu=arm64-v8a,armeabi-v7a,x86_64 //tensorflow/lite/java:tensorflow-lite-gpu')
    copy('bazel-bin/tensorflow/lite/java/tensorflow-lite-gpu.aar', 'Android')

    # GL Delegate
    run_cmd('bazel build -c opt --config=android_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_gl.so')
    copy('bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_gl.so', 'Android/arm64-v8a')
    run_cmd('bazel build -c opt --config=android_arm --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_gl.so')
    copy('bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_gl.so', 'Android/armeabi-v7a')

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
    parser.add_argument('-xnnpack', action = "store_true", default = True,
                        help = 'Build with XNNPACK')

    args = parser.parse_args()
    TENSORFLOW_PATH = os.path.abspath(args.tfpath) 

    platform_name = platform.system()

    if args.macos:
        assert platform_name == 'Darwin', f'-macos not supported on the platform: {platform_name}'
        print('Build macOS')
        build_mac(args.xnnpack)
    
    if args.windows:
        assert platform_name == 'Windows', f'-windows not supported on the platform: {platform_name}'
        print('Build Windows')
        build_windows(args.xnnpack)
    
    if args.linux:
        assert platform_name == 'Linux', f'-linux not supported on the platform: {platform_name}'
        print('Build Linux')
        build_linux(args.xnnpack)
    
    if args.ios:
        assert platform_name == 'Darwin', f'-ios not supported on the platform: {platform_name}'
        # Need to set iOS build option in ./configure
        print('Build iOS')
        build_ios()
    
    if args.android:
        # Need to set Android build option in ./configure
        print('Build Android')
        build_android(args.xnnpack)
