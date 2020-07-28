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

using TfLiteInterpreter = System.IntPtr;
using TfLiteInterpreterOptions = System.IntPtr;
using TfLiteRegistration = System.IntPtr;
using TfLiteContext = System.IntPtr;
using TfLiteNode = System.IntPtr;
using TfLiteModel = System.IntPtr;
using TfLiteTensor = System.IntPtr;
using TfLiteDelegate = System.IntPtr;

namespace TensorFlowLite
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Registration
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void InitDelegate(TfLiteContext context, IntPtr buffer, UInt64 length);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void FreeDelegate(TfLiteContext context, IntPtr buffer);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate Interpreter.Status PrepareDelegate(TfLiteContext context, TfLiteNode node);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate Interpreter.Status InvokeDelegate(TfLiteContext context, TfLiteNode node);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate string ProfilingStringDelegate(TfLiteContext context, TfLiteNode node);

        public InitDelegate initDelegate;
        public FreeDelegate freeDelegate;
        public PrepareDelegate prepareDelegate;
        public InvokeDelegate invokeDelegate;
        public ProfilingStringDelegate profilingStringDelegate;
        public Int32 builtinCode;
        public string customName;
        public int version;
    }

    public static class InterpreterExtension
    {
        public static void AddCustomOp(this Interpreter interpreter,string name, Registration registration)
        {
            // UnityEngine.Debug.Log("add custom op");
            // TfLiteInterpreterOptionsAddCustomOp(interpreter, "HOGE");
        }

#if UNITY_IOS && !UNITY_EDITOR
        private const string TensorFlowLibrary = "__Internal";
#else
        private const string TensorFlowLibrary = "libtensorflowlite_c";
#endif

        [DllImport(TensorFlowLibrary)]
        private static extern Interpreter.Status TfLiteInterpreterResetVariableTensors(TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern void TfLiteInterpreterOptionsAddBuiltinOp(
            TfLiteInterpreterOptions options,
            BuiltinOperator op,
            TfLiteRegistration registration,
            UInt32 min_version, UInt32 max_version);

        [DllImport(TensorFlowLibrary)]
        private static extern void TfLiteInterpreterOptionsAddCustomOp(
            TfLiteInterpreterOptions options,
            string name,
            TfLiteRegistration registration,
            UInt32 min_version, UInt32 max_version);

        [DllImport(TensorFlowLibrary)]
        internal static extern void TfLiteInterpreterOptionsSetUseNNAPI(TfLiteInterpreterOptions options, bool enable);

    }
}
