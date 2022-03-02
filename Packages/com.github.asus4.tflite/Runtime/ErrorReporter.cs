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
using Debug = UnityEngine.Debug;
using TfLiteInterpreterOptions = System.IntPtr;

namespace TensorFlowLite
{
    internal static class ErrorReporter
    {
        // void (*reporter)(void* user_data, const char* format, va_list args),
        [UnmanagedFunctionPointer(CallingConvention.Cdecl, SetLastError = true)]
        private delegate void ErrorReporterDelegate(IntPtr userData, string format, IntPtr argsPtrs);

        internal static void ConfigureReporter(TfLiteInterpreterOptions options)
        {
            TfLiteInterpreterOptionsSetErrorReporter(options, OnErrorReporter, IntPtr.Zero);
        }

        [AOT.MonoPInvokeCallback(typeof(ErrorReporterDelegate))]
        private static void OnErrorReporter(IntPtr userData, string format, IntPtr vaList)
        {
            // Marshalling va_list as args.
            // refs:
            // https://github.com/dotnet/runtime/issues/9316
            // https://github.com/jeremyVignelles/va-list-interop-demo

            string report;
#if UNITY_ANDROID && !UNITY_EDITOR
            report = UnityTFLiteStringFormat(format, vaList);
#else
            // TODO: Support arglist for other platforms
            report = format;
#endif
            Debug.LogWarning($"TFLite Warning: {report}");
        }

        #region Externs

        private const string TensorFlowLibrary = Interpreter.TensorFlowLibrary;

        [DllImport(TensorFlowLibrary, CallingConvention = CallingConvention.Cdecl)]
        private static extern void TfLiteInterpreterOptionsSetErrorReporter(
            TfLiteInterpreterOptions options,
            ErrorReporterDelegate errorReporter,
            IntPtr user_data);

#if UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
        private const string LibCLibrary = "libc";

        [DllImport(LibCLibrary, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private extern static int printf(
            [In][MarshalAs(UnmanagedType.LPStr)] string format,
            IntPtr args);

        [DllImport(LibCLibrary, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private extern static int sprintf(
            IntPtr buffer,
            [In][MarshalAs(UnmanagedType.LPStr)] string format,
            IntPtr args);
#endif // UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX

#if UNITY_ANDROID && !UNITY_EDITOR
    private const string HelperLibrary = "__Internal";

    [DllImport(HelperLibrary, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    private extern static string UnityTFLiteStringFormat(string format, IntPtr vaList);
#endif

        #endregion // Externs
    }
}
