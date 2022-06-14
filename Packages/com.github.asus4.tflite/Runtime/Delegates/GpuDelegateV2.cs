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

#if (UNITY_ANDROID && !UNITY_EDITOR) || (UNITY_EDITOR_LINUX || UNITY_STANDALONE_LINUX)

using System;
using System.Runtime.InteropServices;
using UnityEngine;
using TfLiteDelegate = System.IntPtr;

namespace TensorFlowLite
{
    public class GpuDelegateV2 : IBindableDelegate
    {

        /// <summary>
        /// TfLiteGpuInferenceUsage
        /// Encapsulated compilation/runtime tradeoffs.
        /// </summary>
        public enum Usage
        {
            // Delegate will be used only once, therefore, bootstrap/init time should
            // be taken into account.
            FastSingleAnswer = 0,

            // Prefer maximizing the throughput. Same delegate will be used repeatedly on
            // multiple inputs.
            SustainedSpeed = 1,
        }

        /// <summary>
        /// TfLiteGpuInferencePriority
        /// </summary>
        public enum InferencePriority
        {
            Auto = 0,
            MaxPrecision = 1,
            MinLatency = 2,
            MinMemoryUsage = 3,
        }

        /// <summary>
        /// TfLiteGpuExperimentalFlags
        /// </summary>
        [System.Flags]
        public enum ExperimentalFlags
        {
            None = 0,
            // Enables inference on quantized models with the delegate.
            // NOTE: This is enabled in TfLiteGpuDelegateOptionsV2Default.
            EnableQuant = 1 << 0,
            // Enforces execution with the provided backend.
            ClOnly = 1 << 1,
            GlOnly = 1 << 2,
            // Enable serialization of GPU kernels & model data. Speeds up initialization at the cost of space on disk.
            // NOTE: User also needs to set serialization_dir & model_token in TfLiteGpuDelegateOptionsV2
            EnableSerialization = 1 << 3,
        }

        /// <summary>
        /// the Mirror of TfLiteGpuDelegateOptionsV2
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Options
        {
            public int isPrecisionLossAllowed;
            public int inferencePreference;
            public int inferencePriority1;
            public int inferencePriority2;
            public int inferencePriority3;
            public long experimentalFlags;
            public int maxDelegatedPartitions;
            public IntPtr serializationDir; // char*
            public IntPtr modelToken; // char*
        };

        public TfLiteDelegate Delegate { get; private set; }

        public static Options DefaultOptions => TfLiteGpuDelegateOptionsV2Default();

        public GpuDelegateV2()
        {
            Options options = DefaultOptions;
            Delegate = TfLiteGpuDelegateV2Create(ref options);
        }

        public GpuDelegateV2(Options options)
        {
            Delegate = TfLiteGpuDelegateV2Create(ref options);
        }

        public void Dispose()
        {
            TfLiteGpuDelegateV2Delete(Delegate);
            Delegate = TfLiteDelegate.Zero;
        }

        public bool BindBufferToInputTensor(Interpreter interpreter, int index, ComputeBuffer buffer)
        {
            uint bufferID = (uint)buffer.GetNativeBufferPtr().ToInt32();
            var status = TfLiteGpuDelegateV2BindInputBuffer(Delegate, index, bufferID);
            return status == Interpreter.Status.Ok;
        }

        public bool BindBufferToOutputTensor(Interpreter interpreter, int index, ComputeBuffer buffer)
        {
            uint bufferID = (uint)buffer.GetNativeBufferPtr().ToInt32();
            var status = TfLiteGpuDelegateV2BindOutputBuffer(Delegate, index, bufferID);
            return status == Interpreter.Status.Ok;
        }

#region Externs
#if (UNITY_ANDROID && !UNITY_EDITOR)
        private const string TensorFlowLibraryGPU = "libtensorflowlite_gpu_jni.so";
#else
        private const string TensorFlowLibraryGPU = "libtensorflowlite_gpu_delegate";
#endif

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe Options TfLiteGpuDelegateOptionsV2Default();

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe TfLiteDelegate TfLiteGpuDelegateV2Create(ref Options options);

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe void TfLiteGpuDelegateV2Delete(TfLiteDelegate gpuDelegate);

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe Interpreter.Status TfLiteGpuDelegateBindBufferToTensor(
            TfLiteDelegate gpuDelegate, uint buffer, int tensor_index);

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe Interpreter.Status TfLiteGpuDelegateV2BindInputBuffer(
            TfLiteDelegate gpuDelegatee, int index, uint buffer);

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe Interpreter.Status TfLiteGpuDelegateV2BindOutputBuffer(
            TfLiteDelegate gpuDelegate, int index, uint buffer);

#endregion // Externs
    }
}
#endif // (UNITY_ANDROID && !UNITY_EDITOR) || (UNITY_EDITOR_LINUX || UNITY_STANDALONE_LINUX)

