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
#if UNITY_ANDROID && !UNITY_EDITOR

    /// <summary>
    /// the Mirror of TfLiteGpuDelegateOptionsV2
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Options
    {
        int isPrecisionLossAllowed;
        int inferencePreference;
        int inferencePriority1;
        int inferencePriority2;
        int inferencePriority3;
        long experimentalFlags;
        int maxDelegatedPartitions;
    };

    public class GlDelegate : IGpuDelegate
    {
        public TfLiteDelegate Delegate { get; private set; }

        public GlDelegate()
        {
            Options options = TfLiteGpuDelegateOptionsV2Default();
            Delegate = TfLiteGpuDelegateV2Create(ref options);
        }

        public void Dispose()
        {
            TfLiteGpuDelegateV2Delete(Delegate);
            Delegate = TfLiteDelegate.Zero;
        }

    #region Externs
        private const string TensorFlowLibraryGPU = "libtensorflowlite_gpu_delegate";

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe Options TfLiteGpuDelegateOptionsV2Default();

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe TfLiteDelegate TfLiteGpuDelegateV2Create(ref Options options);

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe void TfLiteGpuDelegateV2Delete(TfLiteDelegate gpuDelegate);
    #endregion // Externs
    }
#endif // UNITY_ANDROID && !UNITY_EDITOR
}
