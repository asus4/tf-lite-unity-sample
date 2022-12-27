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
using TfLiteDelegate = System.IntPtr;
using TfLiteInterpreterOptions = System.IntPtr;

namespace TensorFlowLite
{
    public class InterpreterOptions : IDisposable
    {
        // void (*reporter)(void* user_data, const char* format, va_list args),
        [UnmanagedFunctionPointer(CallingConvention.Cdecl, SetLastError = true)]
        private delegate void ErrorReporterDelegate(IntPtr userData, string format, IntPtr argsPtrs);

        internal TfLiteInterpreterOptions nativePtr;

        private List<IDelegate> delegates;

        private int _threads;
        public int threads
        {
            get => _threads;
            set
            {
                _threads = value;
                TfLiteInterpreterOptionsSetNumThreads(nativePtr, value);
            }
        }

        private bool _useNNAPI;
        [Obsolete("useNNAPI option is deprecated, use NNAPIDelegate instead.")]
        public bool useNNAPI
        {
            get => _useNNAPI;
            set
            {
                _useNNAPI = value;
#if UNITY_ANDROID && !UNITY_EDITOR
                // Create NNAPI delegate with default options
                AddDelegate(new NNAPIDelegate());
#endif // UNITY_ANDROID && !UNITY_EDITOR
            }
        }

        public InterpreterOptions()
        {
            nativePtr = TfLiteInterpreterOptionsCreate();
            delegates = new List<IDelegate>();

            ErrorReporter.ConfigureReporter(nativePtr);
        }

        public void Dispose()
        {
            if (nativePtr != IntPtr.Zero)
            {
                TfLiteInterpreterOptionsDelete(nativePtr);
                nativePtr = IntPtr.Zero;
            }
            foreach (var gpuDelegate in delegates)
            {
                gpuDelegate.Dispose();
            }
            delegates.Clear();
        }

        public void AddDelegate(IDelegate iDelegate)
        {
            if (iDelegate == null) return;
            TfLiteInterpreterOptionsAddDelegate(nativePtr, iDelegate.Delegate);
            delegates.Add(iDelegate);
        }

        public void AddGpuDelegate()
        {
            AddDelegate(CreateGpuDelegate());
        }

#pragma warning disable CS0162 // Unreachable code detected 
        private static IDelegate CreateGpuDelegate()
        {
#if (UNITY_ANDROID && !UNITY_EDITOR) || (UNITY_EDITOR_LINUX || UNITY_STANDALONE_LINUX)
            return new GpuDelegateV2();
#elif UNITY_IOS || UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
            return new MetalDelegate(new MetalDelegate.Options()
            {
                allowPrecisionLoss = false,
                waitType = MetalDelegate.WaitType.Passive,
                enableQuantization = true,
            });
#endif
            UnityEngine.Debug.LogWarning("GPU Delegate is not supported on this platform");
            return null;
        }
#pragma warning restore CS0162 // Unreachable code detected    


        #region Externs
        private const string TensorFlowLibrary = Interpreter.TensorFlowLibrary;

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreterOptions TfLiteInterpreterOptionsCreate();

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions options);

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsSetNumThreads(
            TfLiteInterpreterOptions options,
            int num_threads
        );

        [DllImport(TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsAddDelegate(
            TfLiteInterpreterOptions options,
            TfLiteDelegate _delegate);

        #endregion // Externs
    }
}
