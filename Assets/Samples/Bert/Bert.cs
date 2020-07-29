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

        Dictionary<string, int> vocabularyTable;

        public Bert(string modelPath, string vocabText)
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

            vocabularyTable = LoadVocabularies(vocabText);
        }

        public void Dispose()
        {
            interpreter?.Dispose();
        }

        public void Invoke(string query, string content)
        {
            ToInputs(query, content);

            interpreter.SetInputTensorData(0, inputs0);
            interpreter.SetInputTensorData(1, inputs1);
            interpreter.SetInputTensorData(2, inputs2);

            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, outputs0);
            interpreter.GetOutputTensorData(1, outputs1);
        }

        void ToInputs(string query, string content)
        {
            var tokens = BertTokenizer.BasicTokenize(query);
        }

        public static Dictionary<string, int> LoadVocabularies(string text)
        {
            var lines = text.Split('\n');
            var vocablaries = new Dictionary<string, int>();
            for (int i = 0; i < lines.Length; i++)
            {
                vocablaries.Add(lines[i], i);
            }
            return vocablaries;
        }

    }
}
