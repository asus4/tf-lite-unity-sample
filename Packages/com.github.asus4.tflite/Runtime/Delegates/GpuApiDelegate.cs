/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if (UNITY_ANDROID && !UNITY_EDITOR) || (UNITY_EDITOR_LINUX || UNITY_STANDALONE_LINUX)

using System;
using System.Runtime.InteropServices;
using UnityEngine;
using TfLiteDelegate = System.IntPtr;
using DataType = TensorFlowLite.Interpreter.DataType;

namespace TensorFlowLite
{
    public class GpuApiDelegate : IBindableDelegate
    {
        /// <summary>
        /// the Mirror of TfLiteGpuCompileOptions_New
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct CompileOptions
        {
            // When set to zero, computations are carried out in 32-bit floating point.
            // Otherwise, the GPU may quantify tensors, downcast values, process in FP16
            // (recommended).
            public int precisionLossAllowed;
            // Priority is defined in TfLiteGpuInferencePriority.
            public int inferencePriority;
        };

        [StructLayout(LayoutKind.Sequential)]
        public struct Options
        {
            public CompileOptions compileOptions;

            // [Optional]
            // Whenever EGL display and EGL context are set, corresponding OpenCL context
            // will be created.
            // These variables are required when using GL objects as inputs or outputs.
            public IntPtr eglDisplay;
            public IntPtr eglContext;
            // [Optional]
            // Contains data returned from TfLiteGpuDelegateGetSerializedBinaryCache call.
            // Invalid or incompatible data will be discarded. Compiled binary may become
            // incompatible when GPU driver is updated.
            public IntPtr serializedBinaryCacheData;
            public uint serialized_binary_cache_size;
        }

        public enum DataLayout
        {
            BHWC = 0,
            DHWC4 = 1,
        }

        public TfLiteDelegate Delegate { get; private set; }

        public static Options DefaultOptions => new Options()
        {
            compileOptions = new CompileOptions()
            {
                precisionLossAllowed = 0,
                inferencePriority = (int)GpuDelegateV2.InferencePriority.MinLatency,
            },
            eglDisplay = IntPtr.Zero,
            eglContext = IntPtr.Zero,
            serializedBinaryCacheData = IntPtr.Zero,
            serialized_binary_cache_size = 0,
        };

#if UNITY_ANDROID && !UNITY_EDITOR
        public static Options BindableOptions
        {
            get
            {
                // Call Java API to get EGLDisplay and EGLContext
                var EGL14 = new AndroidJavaClass("android.opengl.EGL14");
                var display = EGL14.CallStatic<AndroidJavaObject>("eglGetCurrentDisplay");
                var context = EGL14.CallStatic<AndroidJavaObject>("eglGetCurrentContext");
                return new Options()
                {
                    compileOptions = new CompileOptions()
                    {
                        precisionLossAllowed = 1,
                        inferencePriority = (int)GpuDelegateV2.InferencePriority.MinLatency,
                    },
                    eglDisplay = display.GetRawObject(),
                    eglContext = context.GetRawObject(),
                    serializedBinaryCacheData = IntPtr.Zero,
                    serialized_binary_cache_size = 0,
                };
            }
        }
#endif

        public GpuApiDelegate()
        {
            Options options = DefaultOptions;
            Delegate = TfLiteGpuDelegateCreate_New(ref options);
        }

        public GpuApiDelegate(Options options)
        {
            Delegate = TfLiteGpuDelegateCreate_New(ref options);
        }

        public void Dispose()
        {
            TfLiteGpuDelegateDelete_New(Delegate);
            Delegate = IntPtr.Zero;
        }

        public bool BindBufferToInputTensor(Interpreter interpreter, int index, ComputeBuffer buffer)
        {
            int tensorIndex = interpreter.GetInputTensorIndex(index);
            return BindBufferToTensor(tensorIndex, buffer);
        }

        public bool BindBufferToOutputTensor(Interpreter interpreter, int index, ComputeBuffer buffer)
        {
            int tensorIndex = interpreter.GetOutputTensorIndex(index);
            return BindBufferToTensor(tensorIndex, buffer);
        }

        private bool BindBufferToTensor(int tensorIndex, ComputeBuffer buffer)
        {
            // TODO: make these configurable
            var dataType = DataType.Float32;
            // var dataLayout = DataLayout.DHWC4;
            var dataLayout = DataLayout.BHWC;

            uint bufferID = (uint)buffer.GetNativeBufferPtr().ToInt32();
            var status = TfLiteGpuDelegateBindGlBufferToTensor(
                Delegate, bufferID, tensorIndex, dataType, dataLayout);
            return status == Interpreter.Status.Ok;
        }

        #region Externs
#if UNITY_ANDROID && !UNITY_EDITOR
        // library name on Android
        private const string TensorFlowLibraryGPU = "libtensorflowlite_gpu_api_delegate.so";
#else
        // library name on Linux
        private const string TensorFlowLibraryGPU = "libtensorflowlite_gpu_delegate";
#endif

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe TfLiteDelegate TfLiteGpuDelegateCreate_New(ref Options options);

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe void TfLiteGpuDelegateDelete_New(TfLiteDelegate gpuDelegate);

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe Interpreter.Status TfLiteGpuDelegateBindGlBufferToTensor(
            TfLiteDelegate gpuDelegate, uint buffer, int tensor_index,
            DataType data_type, DataLayout data_layout);

        #endregion // Externs

    }
}
#endif // (UNITY_ANDROID && !UNITY_EDITOR) || (UNITY_EDITOR_LINUX || UNITY_STANDALONE_LINUX)
