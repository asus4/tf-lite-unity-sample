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
using Status = TensorFlowLite.Interpreter.Status;
using TfLiteContext = System.IntPtr;
using TfLiteDelegate = System.IntPtr;
using TfLiteInterpreter = System.IntPtr;
using TfLiteInterpreterOptions = System.IntPtr;
using TfLiteNode = System.IntPtr;
using TfLiteRegistration = System.IntPtr;
using TfLiteTensor = System.IntPtr;

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
        public delegate Status PrepareDelegate(TfLiteContext context, TfLiteNode node);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate Status InvokeDelegate(TfLiteContext context, TfLiteNode node);
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

    /// <summary>
    /// The bridge for c_api_experimental.h 
    /// </summary>
    public static class InterpreterExperimental
    {
        public static void ResetVariableTensors(this Interpreter interpreter)
        {
            TfLiteInterpreterResetVariableTensors(interpreter.InterpreterPointer);
        }

        public static void AddCustomOp(this Interpreter interpreter, string name, Registration registration)
        {
            throw new NotImplementedException();
            // TfLiteInterpreterOptionsAddCustomOp(interpreter, "HOGE");
        }

        public static void SetAllowBufferHandleOutput(this Interpreter interpreter, bool allowBufferHandleOutput)
        {
            TfLiteSetAllowBufferHandleOutput(interpreter.InterpreterPointer, allowBufferHandleOutput);
        }

        public static Status ModifyGraphWithDelegate(this Interpreter interpreter, IDelegate gpuDelegate)
        {
            return TfLiteInterpreterModifyGraphWithDelegate(interpreter.InterpreterPointer, gpuDelegate.Delegate);
        }

        public static int GetInputTensorIndex(this Interpreter interpreter, int index)
        {
            return TfLiteInterpreterGetInputTensorIndex(interpreter.InterpreterPointer, index);
        }

        public static int GetOutputTensorIndex(this Interpreter interpreter, int index)
        {
            return TfLiteInterpreterGetOutputTensorIndex(interpreter.InterpreterPointer, index);
        }

        private const string TensorFlowLibrary = Interpreter.TensorFlowLibrary;

        [DllImport(TensorFlowLibrary)]
        private static extern Status TfLiteInterpreterResetVariableTensors(TfLiteInterpreter interpreter);

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

        [DllImport(TensorFlowLibrary)]
        internal static extern void TfLiteSetAllowBufferHandleOutput(
            TfLiteInterpreter interpreter,
            bool allow_buffer_handle_output);

        [DllImport(TensorFlowLibrary)]
        internal static extern Status TfLiteInterpreterModifyGraphWithDelegate(
            TfLiteInterpreter interpreter, TfLiteDelegate gpuDelegate);

        [DllImport(TensorFlowLibrary)]
        internal static extern int TfLiteInterpreterGetInputTensorIndex(
            TfLiteInterpreter interpreter, int input_index);

        [DllImport(TensorFlowLibrary)]
        internal static extern int TfLiteInterpreterGetOutputTensorIndex(
            TfLiteInterpreter interpreter, int output_index);

    }
}
