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
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using TfLiteInterpreter = System.IntPtr;
using TfLiteInterpreterOptions = System.IntPtr;
using TfLiteModel = System.IntPtr;
using TfLiteTensor = System.IntPtr;

namespace TensorFlowLite
{
    /// <summary>
    /// Simple C# bindings for the experimental TensorFlowLite C API.
    /// </summary>
    public class Interpreter : IDisposable
    {
        public struct TensorInfo
        {
            public string name { get; internal set; }
            public DataType type { get; internal set; }
            public int[] shape { get; internal set; }
            public QuantizationParams quantizationParams { get; internal set; }

            public override string ToString()
            {
                return string.Format("name: {0}, type: {1}, dimensions: {2}, quantizationParams: {3}",
                  name,
                  type,
                  "[" + string.Join(",", shape) + "]",
                  "{" + quantizationParams + "}");
            }
        }

        private TfLiteModel model = IntPtr.Zero;
        private TfLiteInterpreter interpreter = IntPtr.Zero;
        private readonly InterpreterOptions options = null;
        private readonly GCHandle modelDataHandle;
        private readonly Dictionary<int, GCHandle> inputDataHandles = new Dictionary<int, GCHandle>();
        private readonly Dictionary<int, GCHandle> outputDataHandles = new Dictionary<int, GCHandle>();


        internal TfLiteInterpreter InterpreterPointer => interpreter;

        public Interpreter(byte[] modelData) : this(modelData, null) { }

        public Interpreter(byte[] modelData, InterpreterOptions options)
        {
            modelDataHandle = GCHandle.Alloc(modelData, GCHandleType.Pinned);
            IntPtr modelDataPtr = modelDataHandle.AddrOfPinnedObject();
            model = TfLiteModelCreate(modelDataPtr, modelData.Length);
            if (model == IntPtr.Zero) throw new Exception("Failed to create TensorFlowLite Model");

            this.options = options ?? new InterpreterOptions();

            interpreter = TfLiteInterpreterCreate(model, options.nativePtr);
            if (interpreter == IntPtr.Zero) throw new Exception("Failed to create TensorFlowLite Interpreter");
        }


        public virtual void Dispose()
        {
            if (interpreter != IntPtr.Zero)
            {
                TfLiteInterpreterDelete(interpreter);
                interpreter = IntPtr.Zero;
            }

            if (model != IntPtr.Zero)
            {
                TfLiteModelDelete(model);
                model = IntPtr.Zero;
            }

            options?.Dispose();

            foreach (var handle in inputDataHandles.Values)
            {
                handle.Free();
            }
            foreach (var handle in outputDataHandles.Values)
            {
                handle.Free();
            }
            modelDataHandle.Free();
        }

        public virtual void Invoke()
        {
            ThrowIfError(TfLiteInterpreterInvoke(interpreter));
        }

        public int GetInputTensorCount()
        {
            return TfLiteInterpreterGetInputTensorCount(interpreter);
        }

        public void SetInputTensorData(int inputTensorIndex, Array inputTensorData)
        {
            if (!inputDataHandles.TryGetValue(inputTensorIndex, out GCHandle tensorDataHandle))
            {
                tensorDataHandle = GCHandle.Alloc(inputTensorData, GCHandleType.Pinned);
                inputDataHandles.Add(inputTensorIndex, tensorDataHandle);
            }
            IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();
            TfLiteTensor tensor = TfLiteInterpreterGetInputTensor(interpreter, inputTensorIndex);
            ThrowIfError(TfLiteTensorCopyFromBuffer(tensor, tensorDataPtr, Buffer.ByteLength(inputTensorData)));
        }

        public unsafe void SetInputTensorData<T>(int inputTensorIndex, NativeArray<T> inputTensorData) where T : struct
        {
            IntPtr tensorDataPtr = (IntPtr)NativeArrayUnsafeUtility.GetUnsafePtr(inputTensorData);
            TfLiteTensor tensor = TfLiteInterpreterGetInputTensor(interpreter, inputTensorIndex);
            ThrowIfError(TfLiteTensorCopyFromBuffer(
                tensor, tensorDataPtr, inputTensorData.Length * UnsafeUtility.SizeOf<T>()));
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
            if (!outputDataHandles.TryGetValue(outputTensorIndex, out GCHandle tensorDataHandle))
            {
                tensorDataHandle = GCHandle.Alloc(outputTensorData, GCHandleType.Pinned);
                outputDataHandles.Add(outputTensorIndex, tensorDataHandle);
            }
            IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();
            TfLiteTensor tensor = TfLiteInterpreterGetOutputTensor(interpreter, outputTensorIndex);
            ThrowIfError(TfLiteTensorCopyToBuffer(tensor, tensorDataPtr, Buffer.ByteLength(outputTensorData)));
        }

        public TensorInfo GetInputTensorInfo(int index)
        {
            TfLiteTensor tensor = TfLiteInterpreterGetInputTensor(interpreter, index);
            return GetTensorInfo(tensor);
        }

        public TensorInfo GetOutputTensorInfo(int index)
        {
            TfLiteTensor tensor = TfLiteInterpreterGetOutputTensor(interpreter, index);
            return GetTensorInfo(tensor);
        }

        /// <summary>
        /// Returns a string describing version information of the TensorFlow Lite library.
        /// TensorFlow Lite uses semantic versioning.
        /// </summary>
        /// <returns>A string describing version information</returns>
        public static string GetVersion()
        {
            return Marshal.PtrToStringAnsi(TfLiteVersion());
        }

        private static string GetTensorName(TfLiteTensor tensor)
        {
            return Marshal.PtrToStringAnsi(TfLiteTensorName(tensor));
        }

        protected static TensorInfo GetTensorInfo(TfLiteTensor tensor)
        {
            int[] dimensions = new int[TfLiteTensorNumDims(tensor)];
            for (int i = 0; i < dimensions.Length; i++)
            {
                dimensions[i] = TfLiteTensorDim(tensor, i);
            }
            return new TensorInfo()
            {
                name = GetTensorName(tensor),
                type = TfLiteTensorType(tensor),
                shape = dimensions,
                quantizationParams = TfLiteTensorQuantizationParams(tensor),
            };
        }

        protected TfLiteTensor GetInputTensor(int inputTensorIndex)
        {
            return TfLiteInterpreterGetInputTensor(interpreter, inputTensorIndex);
        }

        protected TfLiteTensor GetOutputTensor(int outputTensorIndex)
        {
            return TfLiteInterpreterGetOutputTensor(interpreter, outputTensorIndex);
        }

        protected static void ThrowIfError(Status status)
        {
            switch (status)
            {
                case Status.Ok:
                    return;
                case Status.Error:
                    throw new Exception("TensorFlowLite operation failed.");
                case Status.DelegateError:
                    throw new Exception("TensorFlowLite delegate operation failed.");
                case Status.ApplicationError:
                    throw new Exception("Applying TensorFlowLite delegate operation failed.");
                case Status.DelegateDataNotFound:
                    throw new Exception("Serialized delegate data not being found.");
                case Status.DelegateDataWriteError:
                    throw new Exception("Writing data to delegate failed.");
                case Status.DelegateDataReadError:
                    throw new Exception("Reading data from delegate failed.");
                case Status.UnresolvedOps:
                    throw new Exception("Ops not found.");
                default:
                    throw new Exception($"Unknown TensorFlowLite error: {status}");
            }
        }

        #region Externs

#if UNITY_IOS && !UNITY_EDITOR
        internal const string TensorFlowLibrary = "__Internal";
#elif UNITY_ANDROID && !UNITY_EDITOR
        internal const string TensorFlowLibrary = "libtensorflowlite_jni";
#else
        internal const string TensorFlowLibrary = "libtensorflowlite_c";
#endif

        // TfLiteStatus
        public enum Status
        {
            Ok = 0,
            Error = 1,
            DelegateError = 2,
            ApplicationError = 3,
            DelegateDataNotFound = 4,
            DelegateDataWriteError = 5,
            DelegateDataReadError = 6,
            UnresolvedOps = 7,
        }

        // TfLiteType
        public enum DataType
        {
            NoType = 0,
            Float32 = 1,
            Int32 = 2,
            UInt8 = 3,
            Int64 = 4,
            String = 5,
            Bool = 6,
            Int16 = 7,
            Complex64 = 8,
            Int8 = 9,
            Float16 = 10,
            Float64 = 11,
            Complex128 = 12,
            UInt64 = 13,
            Resource = 14,
            Variant = 15,
            UInt32 = 16,
            UInt16 = 17,
        }

        public struct QuantizationParams
        {
            public float scale;
            public int zeroPoint;

            public override string ToString()
            {
                return string.Format("scale: {0} zeroPoint: {1}", scale, zeroPoint);
            }
        }

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe IntPtr TfLiteVersion();

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreter TfLiteModelCreate(IntPtr model_data, int model_size);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe void TfLiteModelDelete(TfLiteModel model);

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
        private static extern unsafe Status TfLiteInterpreterResizeInputTensor(
            TfLiteInterpreter interpreter,
            int input_index,
            int[] input_dims,
            int input_dims_size);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe Status TfLiteInterpreterAllocateTensors(
            TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe Status TfLiteInterpreterInvoke(TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteInterpreterGetOutputTensorCount(
            TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteTensor TfLiteInterpreterGetOutputTensor(
            TfLiteInterpreter interpreter,
            int output_index);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe DataType TfLiteTensorType(TfLiteTensor tensor);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe int TfLiteTensorNumDims(TfLiteTensor tensor);

        [DllImport(TensorFlowLibrary)]
        private static extern int TfLiteTensorDim(TfLiteTensor tensor, int dim_index);

        [DllImport(TensorFlowLibrary)]
        private static extern uint TfLiteTensorByteSize(TfLiteTensor tensor);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe IntPtr TfLiteTensorName(TfLiteTensor tensor);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe QuantizationParams TfLiteTensorQuantizationParams(TfLiteTensor tensor);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe Status TfLiteTensorCopyFromBuffer(
            TfLiteTensor tensor,
            IntPtr input_data,
            int input_data_size);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe Status TfLiteTensorCopyToBuffer(
            TfLiteTensor tensor,
            IntPtr output_data,
            int output_data_size);

        #endregion
    }
}
