using System.Diagnostics;
using System.Text;

namespace TensorFlowLite
{
    /// <summary>
    /// Extension methods for interpreter
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
            var sb = new StringBuilder();
            int inputCount = interpreter.GetInputTensorCount();
            int outputCount = interpreter.GetOutputTensorCount();
            for (int i = 0; i < inputCount; i++)
            {
                sb.AppendLine($"Input [{i}]: {interpreter.GetInputTensorInfo(i)}");
            }
            sb.AppendLine();
            for (int i = 0; i < outputCount; i++)
            {
                sb.AppendLine($"Output [{i}]: {interpreter.GetOutputTensorInfo(i)}");
            }
            UnityEngine.Debug.Log(sb.ToString());
        }

        /// <summary>
        /// Print the information about the model Inputs/Outputs for debug.
        /// </summary>
        /// <param name="interpreter">An tflite interpreter</param>
        [Conditional("DEVELOPMENT_BUILD"), Conditional("UNITY_EDITOR")]
        public static void LogIOInfo(this SignatureRunner interpreter)
        {
            var sb = new StringBuilder();
            int signatureCount = interpreter.GetSignatureCount();
            for (int i = 0; i < signatureCount; i++)
            {
                sb.AppendLine($"Signature [{i}]: {interpreter.GetSignatureName(i)}");
            }
            sb.AppendLine();

            int inputCount = interpreter.GetInputTensorCount();
            int outputCount = interpreter.GetOutputTensorCount();
            for (int i = 0; i < inputCount; i++)
            {
                sb.AppendLine($"Input [{i}]: {interpreter.GetInputTensorInfo(i)}");
            }
            sb.AppendLine();
            for (int i = 0; i < outputCount; i++)
            {
                sb.AppendLine($"Output [{i}]: {interpreter.GetOutputTensorInfo(i)}");
            }
            UnityEngine.Debug.Log(sb.ToString());
        }
    }
}
