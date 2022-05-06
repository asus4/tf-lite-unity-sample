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
    public sealed class XNNPackDelegate : IGpuDelegate
    {
        [System.Flags]
        public enum Flags : uint
        {
            // Enable XNNPACK acceleration for signed quantized 8-bit inference.
            // This includes operators with channel-wise quantized weights.
            QS8 = 0x00000001,
            // Enable XNNPACK acceleration for unsigned quantized 8-bit inference.
            QU8 = 0x00000002,
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Options
        {
            public int numThreads;
            public Flags flags;
        }

        public TfLiteDelegate Delegate { get; private set; }

        public static Options DefaultOptions => TfLiteXNNPackDelegateOptionsDefault();

        public XNNPackDelegate()
        {
            Options options = DefaultOptions;
            Delegate = TfLiteXNNPackDelegateCreate(ref options);
        }

        public XNNPackDelegate(Options options)
        {
            Delegate = TfLiteXNNPackDelegateCreate(ref options);
        }

        public void Dispose()
        {
            TfLiteXNNPackDelegateDelete(Delegate);
            Delegate = TfLiteDelegate.Zero;
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
        #endregion // Externs
    }
}
