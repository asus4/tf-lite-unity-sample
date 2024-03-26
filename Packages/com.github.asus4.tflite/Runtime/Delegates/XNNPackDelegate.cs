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
using TfLiteDelegate = System.IntPtr;
using TfLiteXNNPackDelegateWeightsCache = System.IntPtr;

namespace TensorFlowLite
{
    public sealed class XNNPackDelegate : IDelegate
    {
        [System.Flags]
        public enum Flags : uint
        {
            // Enable XNNPACK acceleration for signed quantized 8-bit inference.
            // This includes operators with channel-wise quantized weights.
            QS8 = 0x00000001,
            // Enable XNNPACK acceleration for unsigned quantized 8-bit inference.
            QU8 = 0x00000002,
            // Force FP16 inference for FP32 operators.
            FORCE_FP16 = 0x00000004,
            // Enable XNNPACK acceleration for FULLY_CONNECTED operator with dynamic
            //weights.
            DYNAMIC_FULLY_CONNECTED = 0x00000008,
            // Enable XNNPACK acceleration for VAR_HANDLE, READ_VARIABLE, and
            // ASSIGN_VARIABLE operators.
            VARIABLE_OPERATORS = 0x00000010,
            // Enable transient indirection buffer to reduce memory usage in selected
            // operators. Indirection buffer initialization will take place on every
            // inference run, instead of only once during initialization of the operators.
            TRANSIENT_INDIRECTION_BUFFER = 0x00000020,
            // Enable the latest XNNPACK operators and features in the delegate which have
            // not yet been enabled by default.
            ENABLE_LATEST_OPERATORS = 0x00000040,
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Options
        {
            // Number of threads to use in the thread pool.
            // 0 or negative value means no thread pool used.
            public int numThreads;
            public Flags flags;
            // Cache for packed weights, can be shared between multiple instances of
            // delegates.
            public TfLiteXNNPackDelegateWeightsCache weightsCache;
            // Deprecated. Use the flags bitfield with the
            // TFLITE_XNNPACK_DELEGATE_FLAG_VARIABLE_OPERATORS mask.
            [Obsolete("Use the flags bitfield with the TFLITE_XNNPACK_DELEGATE_FLAG_VARIABLE_OPERATORS mask.")]
            public bool handleVariableOps;
            // Enable adaptive optimization for AVX CPUs.
            public bool experimental_adaptive_avx_optimization;
        }

        public TfLiteDelegate Delegate { get; private set; }

        public static Options DefaultOptions => TfLiteXNNPackDelegateOptionsDefault();

        public XNNPackDelegate(): this(DefaultOptions)
        {
        }

        public XNNPackDelegate(Options options)
        {
            UnityEngine.Debug.Log("XNNPackDelegate Created");
            Delegate = TfLiteXNNPackDelegateCreate(ref options);
        }

        public void Dispose()
        {
            TfLiteXNNPackDelegateDelete(Delegate);
            Delegate = TfLiteDelegate.Zero;
        }

        public static XNNPackDelegate DelegateForType(Type inputType)
        {
            Flags flags = 0;
            if (inputType == typeof(sbyte))
            {
                flags = Flags.QS8;
            }
            else if (inputType == typeof(byte))
            {
                flags = Flags.QU8;
            }
            var options = new Options()
            {
                numThreads = UnityEngine.SystemInfo.processorCount,
                flags = flags,
            };
            return new XNNPackDelegate(options);
        }

        #region Externs
        // APIs for XNNPack are included in the core library 
        internal const string TensorFlowLibrary = Interpreter.TensorFlowLibrary;

        // Returns a structure with the default XNNPack delegate options.
        [DllImport(TensorFlowLibrary)]
        private static extern unsafe Options TfLiteXNNPackDelegateOptionsDefault();

        // Creates a new delegate instance that need to be destroyed with
        // `TfLiteXNNPackDelegateDelete` when delegate is no longer used by TFLite.
        // When `options` is set to `nullptr`, the following default values are used:
        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteDelegate TfLiteXNNPackDelegateCreate(ref Options options);

        // Destroys a delegate created with `TfLiteXNNPackDelegateCreate` call.
        [DllImport(TensorFlowLibrary)]
        private static extern unsafe void TfLiteXNNPackDelegateDelete(TfLiteDelegate xnnPackDelegate);

        // Weights Cache is disable due to build error in iOS and Unity 2021 LTS.
        // https://github.com/asus4/tf-lite-unity-sample/issues/261

        // Creates a new weights cache that can be shared with multiple delegate instances.
        // [DllImport(TensorFlowLibrary)]
        // private static extern unsafe TfLiteXNNPackDelegateWeightsCache TfLiteXNNPackDelegateWeightsCacheCreate();        

        // Destroys a weights cache created with `TfLiteXNNPackDelegateWeightsCacheCreate` call.
        // [DllImport(TensorFlowLibrary)]
        // private static extern unsafe void TfLiteXNNPackWeightsCacheDelete(TfLiteXNNPackDelegateWeightsCache cache);
        #endregion // Externs
    }
}
