using System.IO;
using UnityEngine;

namespace TensorFlowLite
{
    public class SSD : System.IDisposable
    {
        public struct Result
        {
            public int classID;
            public float score;
            public Rect rect;
        }

        const int WIDTH = 300;
        const int HEIGHT = 300;
        const int CHANNELS = 3; // RGB

        Interpreter interpreter;
        ComputeShader compute;
        ComputeBuffer inputBuffer;

        // https://www.tensorflow.org/lite/models/object_detection/overview
        sbyte[] inputs = new sbyte[WIDTH * HEIGHT * CHANNELS];
        float[] outputs0 = new float[10 * 4]; // RECTs
        float[] outputs1 = new float[10]; // Classes
        float[] outputs2 = new float[10]; // Scores
        Result[] results = new Result[10];

        public SSD(string modelPath, ComputeShader compute)
        {
            this.compute = compute;

            interpreter = new Interpreter(File.ReadAllBytes(modelPath));
            interpreter.ResizeInputTensor(0, new int[] { 1, HEIGHT, WIDTH, CHANNELS });
            interpreter.AllocateTensors();

            inputBuffer = new ComputeBuffer(WIDTH * HEIGHT * CHANNELS, sizeof(sbyte)); // uint8
            inputs = new sbyte[WIDTH * HEIGHT * CHANNELS];
        }

        public void Dispose()
        {
            interpreter?.Dispose();
            inputBuffer?.Dispose();
        }

        public void Invoke(Texture texture)
        {
            compute.SetTexture(0, "InputTexture", texture);
            compute.SetBuffer(0, "OutputTensor", inputBuffer);
            compute.Dispatch(0, 300 / 10, 300 / 10, 1);

            inputBuffer.GetData(inputs);

            float startSec = Time.realtimeSinceStartup;
            interpreter.SetInputTensorData(0, inputs);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
            interpreter.GetOutputTensorData(1, outputs1);
            interpreter.GetOutputTensorData(2, outputs2);

            float durationMs = (int)((Time.realtimeSinceStartup - startSec) * 1000);
            Debug.Log($"{durationMs} ms");
        }

        public Result[] GetResults()
        {
            for (int i = 0; i < 10; i++)
            {
                float top = outputs0[i * 4];
                float left = outputs0[i * 4 + 1];
                float bottom = outputs0[i * 4 + 2];
                float right = outputs0[i * 4 + 3];

                results[i] = new Result()
                {
                    classID = (int)outputs1[i],
                    score = outputs2[i],
                    rect = new Rect(left, top, right - left, bottom - top),
                };
            }
            return results;
        }
    }

}