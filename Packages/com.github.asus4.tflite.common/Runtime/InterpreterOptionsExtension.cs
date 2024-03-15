using System;
using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// Extension methods for InterpreterOptions
    /// </summary>
    public static class InterpreterOptionsExtension
    {
        /// <summary>
        /// Find the best delegate and add to options.
        /// </summary>
        /// <param name="options">An interpreter options</param>
        /// <param name="delegateType">A desired delegate type</param>
        /// <param name="inputType">A type of model input (float, sbyte etc.)</param>
        public static void AutoAddDelegate(
            this InterpreterOptions options,
            TfLiteDelegateType delegateType,
            Type inputType)
        {
            switch (delegateType)
            {
                case TfLiteDelegateType.NONE:
                    options.threads = SystemInfo.processorCount;
                    break;
                case TfLiteDelegateType.NNAPI:
                    if (Application.platform == RuntimePlatform.Android)
                    {
#if UNITY_ANDROID && !UNITY_EDITOR
                        // Create NNAPI delegate with default options
                        options.AddDelegate(new NNAPIDelegate());
#endif // UNITY_ANDROID && !UNITY_EDITOR
                    }
                    else
                    {
                        Debug.LogError("NNAPI is only supported on Android");
                    }
                    break;
                case TfLiteDelegateType.GPU:
                    options.AddGpuDelegate();
                    break;
                case TfLiteDelegateType.XNNPACK:
                    options.threads = SystemInfo.processorCount;
                    options.AddDelegate(XNNPackDelegate.DelegateForType(inputType));
                    break;
                default:
                    options.Dispose();
                    throw new NotImplementedException();
            }
        }
    }
}
