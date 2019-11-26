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
        TextureToTensor tex2tensor;


        // https://www.tensorflow.org/lite/models/object_detection/overview
        sbyte[,,] inputs = new sbyte[HEIGHT, WIDTH, CHANNELS];

        float[,] outputs0 = new float[10, 4]; // [top, left, bottom, right] * 10
        float[] outputs1 = new float[10]; // Classes
        float[] outputs2 = new float[10]; // Scores
        Result[] results = new Result[10];
        static readonly TextureToTensor.ResizeOptions resizeOptions = new TextureToTensor.ResizeOptions()
        {
            aspectMode = TextureToTensor.AspectMode.Fill,
            flipX = false,
            flipY = true,
            width = WIDTH,
            height = HEIGHT,
        };

        public SSD(string modelPath)
        {
            var options = new Interpreter.Options()
            {
                threads = 2,
                gpuDelegate = new MetalDelegate(new MetalDelegate.TFLGpuDelegateOptions()
                {
                    allow_precision_loss = false,
                    waitType = MetalDelegate.TFLGpuDelegateWaitType.Passive,
                })
            };
            interpreter = new Interpreter(File.ReadAllBytes(modelPath), options);
            interpreter.ResizeInputTensor(0, new int[] { 1, HEIGHT, WIDTH, CHANNELS });
            interpreter.AllocateTensors();

            tex2tensor = new TextureToTensor();
        }

        public void Dispose()
        {
            interpreter?.Dispose();
            tex2tensor?.Dispose();
        }

        public void Invoke(Texture inputTex)
        {
            RenderTexture tex = tex2tensor.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs);

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
                float top = 1f - outputs0[i, 0];
                float left = outputs0[i, 1];
                float bottom = 1f - outputs0[i, 2];
                float right = outputs0[i, 3];

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
