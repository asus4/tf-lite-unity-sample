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
        /// <param name="interpreter">A TFLite Interpreter</param>
        [Conditional("DEVELOPMENT_BUILD"), Conditional("UNITY_EDITOR")]
        public static void LogIOInfo(this Interpreter interpreter)
        {
            var sb = new StringBuilder();
            sb.Append($"Version: {Interpreter.GetVersion()}, ");
            sb.Append($"Extensions: {Interpreter.GetExtensionApisVersion()}, ");
            sb.Append($"Schema: {Interpreter.GetSchemaVersion()}");

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

        /// <summary>
        /// Print the information about the model Inputs/Outputs for debug.
        /// </summary>
        /// <param name="runner">A TFLite SignatureRunner</param>
        [Conditional("DEVELOPMENT_BUILD"), Conditional("UNITY_EDITOR")]
        public static void LogIOInfo(this SignatureRunner runner)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Version: {Interpreter.GetVersion()}");
            sb.AppendLine();

            int signatureCount = runner.GetSignatureCount();
            for (int i = 0; i < signatureCount; i++)
            {
                sb.AppendLine($"Signature [{i}]: {runner.GetSignatureKey(i)}");
            }
            sb.AppendLine();

            int signatureInputCount = runner.GetSignatureInputCount();
            for (int i = 0; i < signatureInputCount; i++)
            {
                string name = runner.GetSignatureInputName(i);
                sb.AppendLine($"Signature Input [{i}]: {name},\t info: {runner.GetSignatureInputInfo(name)}");
            }
            sb.AppendLine();

            int signatureOutputCount = runner.GetSignatureOutputCount();
            for (int i = 0; i < signatureOutputCount; i++)
            {
                string name = runner.GetSignatureOutputName(i);
                sb.AppendLine($"Signature Output [{i}]: {name},\t info: {runner.GetSignatureOutputInfo(name)}");
            }
            UnityEngine.Debug.Log(sb.ToString());
        }

        /// <summary>
        /// Gets total element count in the tensor.
        /// </summary>
        /// <param name="info">A tensor info</param>
        /// <returns>The total count of the element</returns>
        public static int GetElementCount(this Interpreter.TensorInfo info)
        {
            int[] shape = info.shape;
            int total = 1;
            for (int i = 0; i < shape.Length; i++)
            {
                total *= shape[i];
            }

            if (total < 1)
            {
                throw new System.NotSupportedException("Dynamic shape is not supported");
            }
            return total;
        }
    }
}
