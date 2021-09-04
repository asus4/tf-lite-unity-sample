using System.Diagnostics;
using System.Text;

namespace TensorFlowLite
{
    /// <summary>
    /// An utility for tflite interpreter
    /// </summary>
    public static class InterpreterExtension
    {
        /// <summary>
        /// Print the information about the model Inputs/Outputs for debug.
        /// </summary>
        /// <param name="interpreter">An tflite interpreter</param>
        [Conditional("DEVELOPMENT_BUILD"), Conditional("UNITY_EDITOR")]
        public static void LogIOInfo(this Interpreter interpreter)
        {
            int inputCount = interpreter.GetInputTensorCount();
            int outputCount = interpreter.GetOutputTensorCount();
            var sb = new StringBuilder();
            for (int i = 0; i < inputCount; i++)
            {
                sb.AppendFormat("intput {0}: {1}", i, interpreter.GetInputTensorInfo(i));
                sb.AppendLine();
            }
            for (int i = 0; i < outputCount; i++)
            {
                sb.AppendFormat("output {0}: {1}", i, interpreter.GetOutputTensorInfo(i));
                sb.AppendLine();
            }
            UnityEngine.Debug.Log(sb.ToString());
        }
    }
}
