using System;
using System.Collections.Generic;
using UnityEngine;


namespace TensorFlowLite
{

    public class Bert : IDisposable
    {
        Interpreter interpreter;

        int[] inputs0; // input_ids
        int[] inputs1; // input_mask
        int[] inputs2; // segment_ids
        float[] outputs0; // end_logits
        float[] outputs1; // start_logits

        public Bert(string modelPath, string vocabTable)
        {
            // NO GPU
            var options = new InterpreterOptions()
            {
                threads = 2
            };
            interpreter = new Interpreter(FileUtil.LoadFile(modelPath), options);
            interpreter.LogIOInfo();

            const int BUFFER_SIZE = 384;
            inputs0 = new int[BUFFER_SIZE];
            inputs1 = new int[BUFFER_SIZE];
            inputs2 = new int[BUFFER_SIZE];
            outputs0 = new float[BUFFER_SIZE];
            outputs1 = new float[BUFFER_SIZE];

            interpreter.AllocateTensors();
        }

        public void Dispose()
        {
            interpreter?.Dispose();
        }
    }
}
