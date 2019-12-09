using System;
using System.Collections.Generic;

namespace TensorFlowLite
{

    public class SmartReply : IDisposable
    {
        Interpreter interpreter;

        public SmartReply(string modelPath)
        {
            var options = new Interpreter.Options()
            {
                threads = 2
            };
            interpreter = new Interpreter(FileUtil.LoadFile(modelPath), options);
            interpreter.LogIOInfo();
        }

        public void Dispose()
        {

        }
    }
}
