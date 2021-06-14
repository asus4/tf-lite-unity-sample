using UnityEngine;

namespace TensorFlowLite
{
    public static class InterpreterExtension
    {
        public static void LogIOInfo(this Interpreter interpreter)
        {
            int inputCount = interpreter.GetInputTensorCount();
            int outputCount = interpreter.GetOutputTensorCount();
            var sb = new System.Text.StringBuilder();
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
            Debug.Log(sb.ToString());
        }
    }
}
