/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#if UNITY_ANDROID && !UNITY_EDITOR

using System;
using System.Runtime.InteropServices;
using TfLiteDelegate = System.IntPtr;

namespace TensorFlowLite
{
    public sealed class NNAPIDelegate : IDelegate
    {
        /// <summary>
        /// Preferred Power/perf trade-off. For more details please see
        /// ANeuralNetworksCompilation_setPreference documentation in :
        /// https://developer.android.com/ndk/reference/group/neural-networks.html
        /// </summary>
        [Flags]
        public enum ExecutionPreference : int
        {
            Undefined = -1,
            LowPower = 0,
            FastSingleAnswer = 1,
            SustainedSpeed = 2,
        };

        /// <summary>
        /// The Mirror of TfLiteNnapiDelegateOptions
        /// Use TfLiteNnapiDelegateOptionsDefault() for Default options.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Options
        {
            // Preferred Power/perf trade-off. Default to kUndefined.
            public ExecutionPreference executionPreference;

            // Selected NNAPI accelerator with nul-terminated name.
            // Default to nullptr, which implies the NNAPI default behavior: NNAPI
            // runtime is allowed to use all available accelerators. If the selected
            // accelerator cannot be found, NNAPI will not be used.
            // It is the caller's responsibility to ensure the string is valid for the
            // duration of the Options object lifetime.
            public IntPtr acceleratorName; // char*

            // The nul-terminated cache dir for NNAPI model.
            // Default to nullptr, which implies the NNAPI will not try caching the
            // compilation.
            public IntPtr cacheDir; // char*

            // The unique nul-terminated token string for NNAPI model.
            // Default to nullptr, which implies the NNAPI will not try caching the
            // compilation. It is the caller's responsibility to ensure there is no
            // clash of the tokens.
            // NOTE: when using compilation caching, it is not recommended to use the
            // same delegate instance for multiple models.
            public IntPtr modelToken; // char*

            // Whether to disallow NNAPI CPU usage. Default to 1 (true). Only effective on
            // Android 10 and above. The NNAPI CPU typically performs less well than
            // built-in TfLite kernels, but allowing CPU allows partial acceleration of
            // models. If this is set to true, NNAPI is only used if the whole model is
            // accelerated.
            public int disallowNnapiCpu;

            // Whether to allow fp32 computation to be run in fp16. Default to 0 (false).
            public int allowFp16;

            // Specifies the max number of partitions to delegate. A value <= 0 means
            // no limit. Default to 3.
            // If the delegation of the full set of supported nodes would generate a
            // number of partition greater than this parameter, only
            // <max_number_delegated_partitions> of them will be actually accelerated.
            // The selection is currently done sorting partitions in decreasing order
            // of number of nodes and selecting them until the limit is reached.
            public int maxNumberDelegatedPartitions;

            // The pointer to NNAPI support lib implementation. Default to nullptr.
            // If specified, NNAPI delegate will use the support lib instead of NNAPI in
            // Android OS.
            public IntPtr nnapi_support_library_handle; // void*


            public string CacheDir
            {
                get => Marshal.PtrToStringAuto(cacheDir);
                set
                {
                    cacheDir = Marshal.StringToHGlobalAuto(value);
                }
            }

            public string ModelToken
            {
                get => Marshal.PtrToStringAuto(modelToken);
                set
                {
                    modelToken = Marshal.StringToHGlobalAuto(value);
                }
            }

            public bool AllowFp16
            {
                get => allowFp16 > 0;
                set => allowFp16 = value ? 1 : 0;
            }
        }

        public TfLiteDelegate Delegate { get; private set; }

        public static Options DefaultOptions => TfLiteNnapiDelegateOptionsDefault();

        public NNAPIDelegate() : this(DefaultOptions)
        {
        }

        public NNAPIDelegate(Options options)
        {
            UnityEngine.Debug.Log("NNAPIDelegate Created");
            Delegate = TfLiteNnapiDelegateCreate(ref options);
        }

        public void Dispose()
        {
            if (Delegate != IntPtr.Zero)
            {
                TfLiteNnapiDelegateDelete(Delegate);
                Delegate = IntPtr.Zero;
            }
        }

        #region Externs
        internal const string TensorFlowLibrary = Interpreter.TensorFlowLibrary;

        // Returns a delegate that uses NNAPI for ops execution.
        // Must outlive the interpreter.
        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteDelegate TfLiteNnapiDelegateCreate(ref Options options);

        // Returns TfLiteNnapiDelegateOptions populated with default values.
        [DllImport(TensorFlowLibrary)]
        private static extern unsafe Options TfLiteNnapiDelegateOptionsDefault();

        // Does any needed cleanup and deletes 'delegate'.
        [DllImport(TensorFlowLibrary)]
        private static extern unsafe void TfLiteNnapiDelegateDelete(TfLiteDelegate delegateHandle);

        #endregion // Externs
    }
}

#endif // UNITY_ANDROID && !UNITY_EDITOR
