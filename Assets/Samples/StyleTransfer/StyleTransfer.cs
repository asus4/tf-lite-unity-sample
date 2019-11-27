using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace TensorFlowLite
{
    public class StyleTransfer : System.IDisposable
    {
        Interpreter interpreter;

        public StyleTransfer(string modelPath)
        {
            interpreter = new Interpreter(File.ReadAllBytes(modelPath));
        }

        public void Dispose()
        {
            interpreter?.Dispose();
        }
    }
}