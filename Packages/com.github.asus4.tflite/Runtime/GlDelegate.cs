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


#if UNITY_ANDROID && !UNITY_EDITOR

using System.Runtime.InteropServices;
using UnityEngine;
using TfLiteDelegate = System.IntPtr;

namespace TensorFlowLite
{
    public class GlDelegate : IGpuDelegate
    {
        public enum ObjectType
        {
            FASTEST = 0,
            TEXTURE = 1,
            BUFFER = 2,
        }

        /// <summary>
        /// The Mirror of TfLiteGlCompileOptions
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct CompileOptions
        {
            public int precisionLossAllowed;
            public int preferredGlObjectType;
            public int dynamicBatchEnabled;
            public int inlineParameters;
        }

        /// <summary>
        /// The Mirror of TfLiteGpuDelegateOptions
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Options
        {
            unsafe byte* metadata;
            public CompileOptions compileOptions;
        };

        public TfLiteDelegate Delegate { get; private set; }

        public static Options DefaultOptions => TfLiteGpuDelegateOptionsDefault();

        public GlDelegate(Options options)
        {
            Delegate = TfLiteGpuDelegateCreate(ref options);
        }

        public GlDelegate()
        {
            Options options = DefaultOptions;
            Delegate = TfLiteGpuDelegateCreate(ref options);
        }

        public void Dispose()
        {
            TfLiteGpuDelegateDelete(Delegate);
            Delegate = TfLiteDelegate.Zero;
        }

        public bool BindBufferToTensor(int tensorIndex, ComputeBuffer buffer)
        {
            throw new System.NotImplementedException();
        }

        #region Externs
        private const string TensorFlowLibraryGPU = "libtensorflowlite_gpu_gl";

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe Options TfLiteGpuDelegateOptionsDefault();

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe TfLiteDelegate TfLiteGpuDelegateCreate(ref Options options);

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe void TfLiteGpuDelegateDelete(TfLiteDelegate gpuDelegate);

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe Interpreter.Status TfLiteGpuDelegateBindBufferToTensor(
            TfLiteDelegate gpuDelegate, uint buffer, int tensor_index);
        #endregion // Externs
    }
}
#endif // UNITY_ANDROID && !UNITY_EDITOR

