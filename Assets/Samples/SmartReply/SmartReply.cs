using System;
using System.Collections.Generic;

namespace TensorFlowLite
{

    public class SmartReply : IDisposable
    {
        Interpreter interpreter;
        String[] responses;

        public SmartReply(string modelPath, String[] responses)
        {
            this.responses = responses;

            // No GPU
            var options = new InterpreterOptions()
            {
                threads = 2
            };
            interpreter = new Interpreter(FileUtil.LoadFile(modelPath), options);
            interpreter.LogIOInfo();

            // interpreter.AllocateTensors();
        }

        public void Dispose()
        {
            interpreter?.Dispose();
        }

        public void Invoke(string text)
        {
            interpreter.SetInputTensorData(0, text.ToCharArray());
        }
    }
}
