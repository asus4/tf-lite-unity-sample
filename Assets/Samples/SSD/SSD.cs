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
        uint[] inputInts = new uint[WIDTH * HEIGHT * CHANNELS];
        sbyte[] inputBytes = new sbyte[WIDTH * HEIGHT * CHANNELS];

        float[] outputs0 = new float[10 * 4]; // [top, left, bottom, right] * 10
        float[] outputs1 = new float[10]; // Classes
        float[] outputs2 = new float[10]; // Scores
        Result[] results = new Result[10];

        public SSD(string modelPath, ComputeShader compute)
        {
            this.compute = compute;

            interpreter = new Interpreter(File.ReadAllBytes(modelPath));
            interpreter.ResizeInputTensor(0, new int[] { 1, HEIGHT, WIDTH, CHANNELS });
            interpreter.AllocateTensors();

            inputBuffer = new ComputeBuffer(WIDTH * HEIGHT * CHANNELS, sizeof(uint)); // uint8
        }

        public void Dispose()
        {
            interpreter?.Dispose();
            inputBuffer?.Dispose();
        }

        public void Invoke(Texture texture)
        {
            Debug.Assert(texture.width == WIDTH);
            Debug.Assert(texture.height == HEIGHT);

            compute.SetTexture(0, "InputTexture", texture);
            compute.SetBuffer(0, "OutputTensor", inputBuffer);
            compute.Dispatch(0, WIDTH / 10, HEIGHT / 10, 1);

            // Note:
            // ComputeShader doesn't support byte quantize
            // Therefore receive as uint, then convert to sbyte
            inputBuffer.GetData(inputInts);
            for (int i = 0; i < inputInts.Length; i++)
            {
                inputBytes[i] = (sbyte)inputInts[i];
            }

            Invoke(inputBytes);
        }

        void Invoke(sbyte[] inputs)
        {
            interpreter.SetInputTensorData(0, inputs);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
            interpreter.GetOutputTensorData(1, outputs1);
            interpreter.GetOutputTensorData(2, outputs2);
        }

        public Result[] GetResults()
        {
            for (int i = 0; i < 10; i++)
            {
                // Invert Y to adapt Unity UI space
                float top = 1f - outputs0[i * 4];
                float left = outputs0[i * 4 + 1];
                float bottom = 1f - outputs0[i * 4 + 2];
                float right = outputs0[i * 4 + 3];

                results[i] = new Result()
                {
                    classID = (int)outputs1[i],
                    score = outputs2[i],
                    rect = new Rect(left, top, right - left, top - bottom),
                };
            }
            return results;
        }
    }
}
