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

using System.Runtime.InteropServices;

using TfLiteDelegate = System.IntPtr;

namespace TensorFlowLite
{
#if UNITY_IOS || UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
    /// <summary>
    /// Metal GPU Delegate
    /// Available on iOS or macOS
    /// </summary>
    public class MetalDelegate : IGpuDelegate
    {
        public enum WaitType
        {
            Passive = 0,
            Active = 1,
            DoNotWait = 2,
            Aggressive = 3,
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Options
        {
            public bool allowPrecisionLoss;
            public WaitType waitType;
            public bool enableQuantization;
        }

        public TfLiteDelegate Delegate { get; private set; }

        public MetalDelegate(Options options)
        {
            Delegate = TFLGpuDelegateCreate(ref options);
        }

        public void Dispose()
        {
            TFLGpuDelegateDelete(Delegate);
            Delegate = TfLiteDelegate.Zero;
        }

        #region Externs

#if UNITY_IOS && !UNITY_EDITOR
        private const string TensorFlowLibraryGPU = "__Internal";
#else
        private const string TensorFlowLibraryGPU = "libtensorflowlite_metal_delegate";
#endif // UNITY_IOS && !UNITY_EDITOR

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe TfLiteDelegate TFLGpuDelegateCreate(ref Options delegateOptions);

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe void TFLGpuDelegateDelete(TfLiteDelegate gpuDelegate);

        #endregion
    }
#endif // UNITY_IOS || UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
}
