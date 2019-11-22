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

namespace TensorFlowLite
{
    public abstract class GpuDelegate : IDisposable
    {
        protected TfLiteDelegate gpuDelegate;

        public TfLiteDelegate Delegate
        {
            get
            {
                return gpuDelegate;
            }
        }

        protected GpuDelegate(TfLiteDelegate gpuDelegate)
        {
            this.gpuDelegate = gpuDelegate;
            if (gpuDelegate == IntPtr.Zero)
            {
                throw new Exception("Failed to create TensorFlowLite GPU Delegate");
            }
        }

        public abstract void Dispose();
    }

    public class MetalDelegate : GpuDelegate
    {
        public enum TFLGpuDelegateWaitType
        {
            Passive = 0,
            Active = 1,
            DoNotWait = 2,
            Aggressive = 3,
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct TFLGpuDelegateOptions
        {
            public bool allow_precision_loss;
            public TFLGpuDelegateWaitType waitType;
        }

        public MetalDelegate(TFLGpuDelegateOptions options) : base(TFLGpuDelegateCreate(options))
        {
        }

        public override void Dispose()
        {
            TFLGpuDelegateDelete(gpuDelegate);
        }

        #region Externs

#if UNITY_IPHONE && !UNITY_EDITOR
    private const string TensorFlowLibraryGPU = "__Internal";
#else
        private const string TensorFlowLibraryGPU = "tensorflow_lite_gpu_dylib";
#endif

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe TfLiteDelegate TFLGpuDelegateCreate(TFLGpuDelegateOptions delegateOptions);

        [DllImport(TensorFlowLibraryGPU)]
        private static extern unsafe void TFLGpuDelegateDelete(TfLiteDelegate gpuDelegate);

        #endregion
    }

}