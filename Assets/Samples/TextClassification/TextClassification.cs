using System;
using System.Collections.Generic;

namespace TensorFlowLite
{

    public class TextClassification : IDisposable
    {
        public struct Result
        {
            public float negative;
            public float positive;
        }

        enum Marker
        {
            PAD = 0, // used for padding
            START = 1, // mark for the start of a sentence
            UNKNOWN = 2, //  mark for unknown words (OOV)
        }

        Interpreter interpreter;
        float[] inputs;
        float[] outputs;

        public Result result { get; private set; }

        Dictionary<string, int> vocabulary;

        public TextClassification(string modelPath, string vocabularyText)
        {
            vocabulary = BuildVocabulary(vocabularyText);

            // NO GPU
            var options = new InterpreterOptions()
            {
                threads = 2
            };
            interpreter = new Interpreter(FileUtil.LoadFile(modelPath), options);

            var inputInfo = interpreter.GetInputTensorInfo(0);
            var outputInfo = interpreter.GetOutputTensorInfo(0);
            inputs = new float[inputInfo.shape[1]];
            outputs = new float[outputInfo.shape[1]];
            interpreter.ResizeInputTensor(0, inputInfo.shape);
            interpreter.AllocateTensors();
        }

        public void Dispose()
        {
            interpreter?.Dispose();
        }

        public void Invoke(string text)
        {
            TextToInput(text, inputs);

            interpreter.SetInputTensorData(0, inputs);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs);

            result = new Result()
            {
                negative = outputs[0],
                positive = outputs[1],
            };
        }


        void TextToInput(string text, float[] inputs)
        {
            // Normalize text
            text = text.ToLower();
            char[] separator = { ',', '.', '!', '?', ' ', '\n' };
            string[] words = text.Split(separator, StringSplitOptions.RemoveEmptyEntries);

            // Start
            int index = 0;
            inputs[index] = (float)Marker.START;

            foreach (var word in words)
            {
                index++;
                if (index >= inputs.Length)
                {
                    break;
                }
                int val;
                if (vocabulary.TryGetValue(word, out val))
                {
                    // UnityEngine.Debug.Log($"{word} : {val}");
                    inputs[index] = (float)val;
                }
                else
                {
                    UnityEngine.Debug.Log($"{word} : UNKNOWN");
                    inputs[index] = (float)Marker.UNKNOWN;
                }
            }

            // Fill Padding
            for (; index < inputs.Length; index++)
            {
                inputs[index] = (float)Marker.PAD;
            }
        }

        static Dictionary<string, int> BuildVocabulary(string vocabularyText)
        {
            string[] lines = vocabularyText.Split(new char[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
            var dict = new Dictionary<string, int>();
            foreach (string line in lines)
            {
                string[] words = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                dict.Add(words[0], int.Parse(words[1]));
            }
            return dict;
        }

    }
}
