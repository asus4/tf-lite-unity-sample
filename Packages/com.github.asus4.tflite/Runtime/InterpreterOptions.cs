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

using TfLiteInterpreter = System.IntPtr;
using TfLiteInterpreterOptions = System.IntPtr;
using TfLiteTensor = System.IntPtr;
using TfLiteDelegate = System.IntPtr;

namespace TensorFlowLite
{
    public class InterpreterOptions : IDisposable
    {
        internal TfLiteInterpreterOptions nativePtr;

        private List<IGpuDelegate> delegates;
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

        public InterpreterOptions()
        {
            nativePtr = TfLiteInterpreterOptionsCreate();
            delegates = new List<IGpuDelegate>();
        }

        public void Dispose()
        {
            if (nativePtr != IntPtr.Zero)
            {
                TfLiteInterpreterOptionsDelete(nativePtr);
            }
            foreach (var gpuDelegate in delegates)
            {
                gpuDelegate.Dispose();
            }
            delegates.Clear();
        }

        public void AddGpuDelegate(IGpuDelegate gpuDelegate)
        {
            TfLiteInterpreterOptionsAddDelegate(nativePtr, gpuDelegate.Delegate);
            delegates.Add(gpuDelegate);
        }

        #region Externs

#if UNITY_IOS && !UNITY_EDITOR
        private const string TensorFlowLibrary = "__Internal";
#else
        private const string TensorFlowLibrary = "libtensorflowlite_c";
#endif


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
