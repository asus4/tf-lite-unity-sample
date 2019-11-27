using UnityEngine;

namespace TensorFlowLite
{
    public static class InterpreterExtension
    {
        public static void LogIOInfo(this Interpreter interpreter)
        {
            int inputCount = interpreter.GetInputTensorCount();
            int outputCount = interpreter.GetOutputTensorCount();
            for (int i = 0; i < inputCount; i++)
            {
                Debug.Log(interpreter.GetInputTensorInfo(i));
            }
            for (int i = 0; i < outputCount; i++)
            {
                Debug.Log(interpreter.GetOutputTensorInfo(i));
            }
        }
    }
}
