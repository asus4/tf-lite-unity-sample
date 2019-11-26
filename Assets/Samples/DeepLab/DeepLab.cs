using System.Diagnostics;
using System.IO;
using UnityEngine;

namespace TensorFlowLite
{
    public class DeepLab : System.IDisposable
    {
        // Port from
        // https://github.com/tensorflow/examples/blob/master/lite/examples/image_segmentation/ios/ImageSegmentation/ImageSegmentator.swift
        static readonly Color32[] COLOR_TABLE = new Color32[]
        {
            ToColor(0xFF00_0000), // Black
            ToColor(0xFF80_3E75), // Strong Purple
            ToColor(0xFFFF_6800), // Vivid Orange
            ToColor(0xFFA6_BDD7), // Very Light Blue
            ToColor(0xFFC1_0020), // Vivid Red
            ToColor(0xFFCE_A262), // Grayish Yellow
            ToColor(0xFF81_7066), // Medium Gray
            ToColor(0xFF00_7D34), // Vivid Green
            ToColor(0xFFF6_768E), // Strong Purplish Pink
            ToColor(0xFF00_538A), // Strong Blue
            ToColor(0xFFFF_7A5C), // Strong Yellowish Pink
            ToColor(0xFF53_377A), // Strong Violet
            ToColor(0xFFFF_8E00), // Vivid Orange Yellow
            ToColor(0xFFB3_2851), // Strong Purplish Red
            ToColor(0xFFF4_C800), // Vivid Greenish Yellow
            ToColor(0xFF7F_180D), // Strong Reddish Brown
            ToColor(0xFF93_AA00), // Vivid Yellowish Green
            ToColor(0xFF59_3315), // Deep Yellowish Brown
            ToColor(0xFFF1_3A13), // Vivid Reddish Orange
            ToColor(0xFF23_2C16), // Dark Olive Green
            ToColor(0xFF00_A1C2), // Vivid Blue
        };

        const int WIDTH = 257;
        const int HEIGHT = 257;
        const int CHANNELS = 3; // RGB

        Interpreter interpreter;
        TextureToTensor tex2tensor;

        // https://www.tensorflow.org/lite/models/segmentation/overview
        float[,,] inputs = new float[HEIGHT, WIDTH, CHANNELS];

        float[,,] outputs0 = new float[HEIGHT, WIDTH, 21];

        Color32[] labelPixels = new Color32[WIDTH * HEIGHT];
        Texture2D labelTex;

        Stopwatch stopWatch = new Stopwatch();

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
            labelTex = new Texture2D(WIDTH, HEIGHT, TextureFormat.ARGB32, 0, false);

            UnityEngine.Debug.LogFormat("tables :{0}", COLOR_TABLE.Length);
        }

        public void Dispose()
        {
            interpreter?.Dispose();
            tex2tensor?.Dispose();
            if (labelTex != null)
            {
                Object.Destroy(labelTex);
            }
        }

        public void Invoke(Texture inputTex)
        {
            RenderTexture tex = tex2tensor.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs);

            interpreter.SetInputTensorData(0, inputs);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
        }

        public Texture2D GetResultTexture()
        {
            stopWatch.Restart();

            int rows = outputs0.GetLength(0); // y
            int cols = outputs0.GetLength(1); // x
            int labels = outputs0.GetLength(2);

            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    int argmax = ArgMaxZ(outputs0, y, x, labels);
                    labelPixels[y * cols + x] = COLOR_TABLE[argmax];
                }
            }

            labelTex.SetPixels32(labelPixels);
            labelTex.Apply();

            stopWatch.Stop();

            UnityEngine.Debug.LogFormat("{0:0.0000}", stopWatch.ElapsedMilliseconds);
            return labelTex;
        }

        public static int ArgMaxZ(float[,,] arr, int x, int y, int numZ)
        {
            // Argmax
            int maxIndex = -1;
            float maxScore = float.MinValue;
            for (int z = 0; z < numZ; z++)
            {
                if (arr[x, y, z] > maxScore)
                {
                    maxScore = arr[x, y, z];
                    maxIndex = z;
                }
            }
            return maxIndex;
        }

        public static Color32 ToColor(uint aCol)
        {
            return new Color32()
            {
                b = (byte)((aCol) & 0xFF),
                g = (byte)((aCol >> 8) & 0xFF),
                r = (byte)((aCol >> 16) & 0xFF),
                a = (byte)((aCol >> 24) & 0xFF),
            };
        }


    }
}