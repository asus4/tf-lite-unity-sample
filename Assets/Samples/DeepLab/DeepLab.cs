using System.IO;
using UnityEngine;

namespace TensorFlowLite
{
    public class DeepLab : System.IDisposable
    {
      
        const int WIDTH = 257;
        const int HEIGHT = 257;
        const int CHANNELS = 3; // RGB

        Interpreter interpreter;
        TextureToTensor tex2tensor;

        // https://www.tensorflow.org/lite/models/segmentation/overview
        float[,,] inputs = new float[HEIGHT, WIDTH, CHANNELS];

        float[,,] outputs0 = new float[HEIGHT, WIDTH, 21];

        static readonly TextureToTensor.ResizeOptions resizeOptions = new TextureToTensor.ResizeOptions()
        {
            aspectMode = TextureToTensor.AspectMode.Fill,
            flipX = false,
            flipY = true,
            width = WIDTH,
            height = HEIGHT,
        };

        public DeepLab(string modelPath)
        {
            interpreter = new Interpreter(File.ReadAllBytes(modelPath), 2);
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
        }

    
    }
}