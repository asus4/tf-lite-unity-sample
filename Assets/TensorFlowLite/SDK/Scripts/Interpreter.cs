/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

using TfLiteInterpreter = System.IntPtr;
using TfLiteInterpreterOptions = System.IntPtr;
using TfLiteModel = System.IntPtr;
using TfLiteTensor = System.IntPtr;
using TfLiteDelegate = System.IntPtr;

namespace TensorFlowLite
{
    /// <summary>
    /// Simple C# bindings for the experimental TensorFlowLite C API.
    /// </summary>
    public class Interpreter : IDisposable
    {
#if UNITY_IPHONE && !UNITY_EDITOR
        private const string TensorFlowLibrary = "__Internal";
        private const string TensorFlowLibraryGPU = "__Internal";
#else
        private const string TensorFlowLibrary = "libtensorflowlite_c";
        private const string TensorFlowLibraryGPU = "tensorflow_lite_gpu_metal";
#endif

        private TfLiteModel model;
        private TfLiteInterpreter interpreter;
        private TfLiteInterpreterOptions interpreterOptions;

        private TfLiteDelegate gpuDelegate;


        public Interpreter(byte[] modelData)
        {
            GCHandle modelDataHandle = GCHandle.Alloc(modelData, GCHandleType.Pinned);
            IntPtr modelDataPtr = modelDataHandle.AddrOfPinnedObject();
            model = TfLiteModelCreate(modelDataPtr, modelData.Length);
            if (model == IntPtr.Zero) throw new Exception("Failed to create TensorFlowLite Model");

            interpreterOptions = TfLiteInterpreterOptionsCreate();
            const int NUM_THREADS = 2;
            TfLiteInterpreterOptionsSetNumThreads(interpreterOptions, NUM_THREADS);

            CreateGpuDelegate();

            interpreter = TfLiteInterpreterCreate(model, interpreterOptions);
            if (interpreter == IntPtr.Zero) throw new Exception("Failed to create TensorFlowLite Interpreter");
        }

        public void Dispose()
        {
            if (interpreter != IntPtr.Zero)
            {
                TfLiteInterpreterDelete(interpreter);
            }
            interpreter = IntPtr.Zero;

            if (model != IntPtr.Zero)
            {
                TfLiteModelDelete(model);
            }
            model = IntPtr.Zero;

            if (interpreterOptions != IntPtr.Zero)
            {
                TfLiteInterpreterOptionsDelete(interpreterOptions);
            }
            interpreterOptions = IntPtr.Zero;

            if (gpuDelegate != IntPtr.Zero)
            {
                TFLGpuDelegateDelete(gpuDelegate);
                UnityEngine.Debug.Log("hoghogehoge");
            }
            gpuDelegate = IntPtr.Zero;
        }

        void CreateGpuDelegate()
        {
            var glCompileOptions = new TfLiteGlCompileOptions();
            glCompileOptions.precision_loss_allowed = 0;
            glCompileOptions.preferred_gl_object_type = (Int32)TfLiteGlObjectType.TFLITE_GL_OBJECT_TYPE_FASTEST;
            glCompileOptions.dynamic_batch_enabled = 0;
            glCompileOptions.inline_parameters = 1;

            var gpuDelegateOptions = new TfLiteGpuDelegateOptions();
            gpuDelegateOptions.metadata = IntPtr.Zero;
            gpuDelegateOptions.compile_options = glCompileOptions;

            gpuDelegate = TFLGpuDelegateCreate(gpuDelegateOptions);
            if (gpuDelegate == IntPtr.Zero)
            {
                throw new Exception("TensorFlowLite GPU create failed.");
            }
            TfLiteInterpreterOptionsAddDelegate(interpreterOptions, gpuDelegate);
        }

        public void Invoke()
        {
            ThrowIfError(TfLiteInterpreterInvoke(interpreter));
        }

        public int GetInputTensorCount()
        {
            return TfLiteInterpreterGetInputTensorCount(interpreter);
        }

        public void SetInputTensorData(int inputTensorIndex, Array inputTensorData)
        {
            GCHandle tensorDataHandle = GCHandle.Alloc(inputTensorData, GCHandleType.Pinned);
            IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();
            TfLiteTensor tensor = TfLiteInterpreterGetInputTensor(interpreter, inputTensorIndex);
            ThrowIfError(TfLiteTensorCopyFromBuffer(
                tensor, tensorDataPtr, Buffer.ByteLength(inputTensorData)));
        }

        public unsafe void SetInputTensorData<T>(int inputTensorIndex, NativeArray<T> inputTensorData) where T : struct
        {
            IntPtr tensorDataPtr = new IntPtr(NativeArrayUnsafeUtility.GetUnsafePtr(inputTensorData));
            TfLiteTensor tensor = TfLiteInterpreterGetInputTensor(interpreter, inputTensorIndex);
            ThrowIfError(TfLiteTensorCopyFromBuffer(
                tensor, tensorDataPtr, inputTensorData.Length));
        }

        public void ResizeInputTensor(int inputTensorIndex, int[] inputTensorShape)
        {
            ThrowIfError(TfLiteInterpreterResizeInputTensor(
                interpreter, inputTensorIndex, inputTensorShape, inputTensorShape.Length));
        }

        public void AllocateTensors()
        {
            ThrowIfError(TfLiteInterpreterAllocateTensors(interpreter));
        }

        public int GetOutputTensorCount()
        {
            return TfLiteInterpreterGetOutputTensorCount(interpreter);
        }

        public void GetOutputTensorData(int outputTensorIndex, Array outputTensorData)
        {
            GCHandle tensorDataHandle = GCHandle.Alloc(outputTensorData, GCHandleType.Pinned);
            IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();
            TfLiteTensor tensor = TfLiteInterpreterGetOutputTensor(interpreter, outputTensorIndex);
            ThrowIfError(TfLiteTensorCopyToBuffer(
                tensor, tensorDataPtr, Buffer.ByteLength(outputTensorData)));
        }

        public unsafe void GetOutputTensorData<T>(int outputTensorIndex, NativeArray<T> outputTensorData) where T : struct
        {
            IntPtr tensorDataPtr = new IntPtr(NativeArrayUnsafeUtility.GetUnsafePtr(outputTensorData));
            TfLiteTensor tensor = TfLiteInterpreterGetOutputTensor(interpreter, outputTensorIndex);
            ThrowIfError(TfLiteTensorCopyToBuffer(
                tensor, tensorDataPtr, outputTensorData.Length));
        }

        private static void ThrowIfError(int resultCode)
        {
            if (resultCode != 0) throw new Exception("TensorFlowLite operation failed.");
        }

        #region Externs
        public enum TfLiteStatus { kTfLiteOk = 0, kTfLiteError = 1 };
        public enum TfLiteType
        {
            kTfLiteNoType = 0, kTfLiteFloat32 = 1, kTfLiteInt32 = 2, kTfLiteUInt8 = 3, kTfLiteInt64 = 4,
            kTfLiteString = 5, kTfLiteBool = 6, kTfLiteInt16 = 7, kTfLiteComplex64 = 8, kTfLiteInt8 = 9, kTfLiteFloat16 = 10,
        };
        public struct TfLiteQuantizationParams
        {
            public float scale;
            public Int32 zero_point;
        };

        public enum TfLiteGlObjectType
        {
            TFLITE_GL_OBJECT_TYPE_FASTEST = 0,
            TFLITE_GL_OBJECT_TYPE_TEXTURE = 1,
            TFLITE_GL_OBJECT_TYPE_BUFFER = 2,
        };
        public struct TfLiteGlCompileOptions
        {
            public Int32 precision_loss_allowed;
            public Int32 preferred_gl_object_type;
            public Int32 dynamic_batch_enabled;
            public Int32 inline_parameters;
        };
        public unsafe struct TfLiteGpuDelegateOptions
        {
            public IntPtr metadata;
            public TfLiteGlCompileOptions compile_options;
        };

        public enum TfLiteDelegateFlags
        {
            kTfLiteDelegateFlagsNone = 0,
            kTfLiteDelegateFlagsAllowDynamicTensors = 1
        };

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreter TfLiteModelCreate(IntPtr model_data, int model_size);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreter TfLiteModelDelete(TfLiteModel model);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreter TfLiteInterpreterCreate(
            TfLiteModel model,
            TfLiteInterpreterOptions optional_options);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterDelete(TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteInterpreterGetInputTensorCount(
            TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteTensor TfLiteInterpreterGetInputTensor(
            TfLiteInterpreter interpreter,
            int input_index);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteInterpreterResizeInputTensor(
            TfLiteInterpreter interpreter,
            int input_index,
            int[] input_dims,
            int input_dims_size);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteInterpreterAllocateTensors(
            TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteInterpreterInvoke(TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteInterpreterGetOutputTensorCount(
            TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteTensor TfLiteInterpreterGetOutputTensor(
            TfLiteInterpreter interpreter,
            int output_index);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteTensorCopyFromBuffer(
            TfLiteTensor tensor,
            IntPtr input_data,
            int input_data_size);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteTensorCopyToBuffer(
            TfLiteTensor tensor,
            IntPtr output_data,
            int output_data_size);


        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreterOptions TfLiteInterpreterOptionsCreate();

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsAddDelegate(TfLiteInterpreterOptions options, TfLiteDelegate delegate_);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions options);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions options, Int32 numThreads);


        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe TfLiteDelegate TFLGpuDelegateCreate(TfLiteGpuDelegateOptions delegateOptions);

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe void TFLGpuDelegateDelete(TfLiteDelegate delegate_);
        #endregion
    }
}
