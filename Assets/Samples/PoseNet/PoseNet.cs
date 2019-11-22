using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using UnityEngine;
using Unity.Mathematics;

namespace TensorFlowLite
{

    /// <summary>
    /// Pose Estimation Example
    /// https://www.tensorflow.org/lite/models/pose_estimation/overview
    /// </summary>
    public class PoseNet : System.IDisposable
    {
        public enum Part
        {
            NOSE,
            LEFT_EYE,
            RIGHT_EYE,
            LEFT_EAR,
            RIGHT_EAR,
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_ELBOW,
            RIGHT_ELBOW,
            LEFT_WRIST,
            RIGHT_WRIST,
            LEFT_HIP,
            RIGHT_HIP,
            LEFT_KNEE,
            RIGHT_KNEE,
            LEFT_ANKLE,
            RIGHT_ANKLE
        }

        public static readonly Part[,] Connections = new Part[,] {
            // HEAD
            { Part.LEFT_EAR, Part.LEFT_EYE },
            { Part.LEFT_EYE, Part.NOSE },
            { Part.NOSE, Part.RIGHT_EYE },
            { Part.RIGHT_EYE, Part.RIGHT_EAR },
            // BODY
            { Part.LEFT_HIP, Part.LEFT_SHOULDER },
            { Part.LEFT_ELBOW, Part.LEFT_SHOULDER },
            { Part.LEFT_ELBOW, Part.LEFT_WRIST },
            { Part.LEFT_HIP, Part.LEFT_KNEE },
            { Part.LEFT_KNEE, Part.LEFT_ANKLE },
            { Part.RIGHT_HIP, Part.RIGHT_SHOULDER },
            { Part.RIGHT_ELBOW, Part.RIGHT_SHOULDER },
            { Part.RIGHT_ELBOW, Part.RIGHT_WRIST },
            { Part.RIGHT_HIP, Part.RIGHT_KNEE },
            { Part.RIGHT_KNEE, Part.RIGHT_ANKLE },
            { Part.LEFT_SHOULDER, Part.RIGHT_SHOULDER },
            { Part.LEFT_HIP, Part.RIGHT_HIP }
        };

        [System.Serializable]
        public struct Result
        {
            public Part part;
            public float confidence;
            public float x;
            public float y;
        }

        const int WIDTH = 257;
        const int HEIGHT = 257;
        const int CHANNELS = 3; // RGB


        Interpreter interpreter;
        TextureToTensor tex2tensor;
        Result[] results = new Result[17];

        float[,,] inputs = new float[WIDTH, HEIGHT, CHANNELS];
        float[,,] outputs0 = new float[9, 9, 17]; // heatmap
        float[,,] outputs1 = new float[9, 9, 34]; // offset
        // float[] outputs2 = new float[9 * 9 * 32]; // displacement fwd
        // float[] outputs3 = new float[9 * 9 * 32]; // displacement bwd

        public float[,,] heatmap => outputs0;
        public float[,,] offsets => outputs1;
        public Vector3[] posisions;

        public PoseNet(string modelPath)
        {
            GpuDelegate gpu = null;
            gpu = new MetalDelegate(new MetalDelegate.TFLGpuDelegateOptions()
            {
                allow_precision_loss = false,
                waitType = MetalDelegate.TFLGpuDelegateWaitType.Passive,
            });

            interpreter = new Interpreter(File.ReadAllBytes(modelPath), 2, gpu);
            interpreter.ResizeInputTensor(0, new int[] { 1, HEIGHT, WIDTH, CHANNELS });
            interpreter.AllocateTensors();

            int inputs = interpreter.GetInputTensorCount();
            int outputs = interpreter.GetOutputTensorCount();
            for (int i = 0; i < inputs; i++)
            {
                Debug.Log(interpreter.GetInputTensorInfo(i));
            }
            for (int i = 0; i < outputs; i++)
            {
                Debug.Log(interpreter.GetOutputTensorInfo(i));
            }

            tex2tensor = new TextureToTensor();
        }

        public void Dispose()
        {
            interpreter?.Dispose();
            tex2tensor?.Dispose();
        }

        public void Invoke(Texture inputTex)
        {
            RenderTexture tex = tex2tensor.Resize(inputTex, WIDTH, HEIGHT);
            tex2tensor.ToTensor(tex, inputs);

            interpreter.SetInputTensorData(0, inputs);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
            interpreter.GetOutputTensorData(1, outputs1);
            // not using
            // interpreter.GetOutputTensorData(2, outputs2);
            // interpreter.GetOutputTensorData(3, outputs3);
        }

        public Result[] GetResults()
        {
            // Name alias
            float[,,] scores = outputs0;
            float[,,] offsets = outputs1;

            // x, y, score
            int ROWS = scores.GetLength(0); //y
            int COLS = scores.GetLength(1); //x
            int PARTS = scores.GetLength(2);

            ApplySigmoid(scores);

            // argmax2d
            // Reset Keypoints
            Vector3[] posisions = new Vector3[PARTS];
            for (int i = 0; i < posisions.Length; i++)
            {
                posisions[i] = new Vector3(-1, -1, float.MinValue);
            }

            for (int y = 0; y < ROWS; y++)
            {
                for (int x = 0; x < COLS; x++)
                {
                    for (int part = 0; part < PARTS; part++)
                    {
                        float confidence = scores[y, x, part];
                        if (confidence > posisions[part].z)
                        {
                            posisions[part] = new Vector3(x, y, confidence);
                        }
                    }
                }
            }
            this.posisions = posisions;

            const int STRIDE = 9 - 1;
            for (int part = 0; part < results.Length; part++)
            {
                int y = (int)posisions[part].y;
                int x = (int)posisions[part].x;
                float offsetX = offsets[y, x, part * 2];
                float offsetY = offsets[y, x, part * 2 + 1];
                results[part] = new Result()
                {
                    part = (Part)part,
                    x = ((float)x / STRIDE * WIDTH + offsetX) / WIDTH,
                    y = ((float)y / STRIDE * HEIGHT + offsetY) / HEIGHT,
                    confidence = (int)posisions[part].z,
                };
            }

            return results;
        }

        static float Sigmoid(float x)
        {
            return (1.0f / (1.0f + Mathf.Exp(-x)));
        }

        static void ApplySigmoid(float[,,] arr)
        {
            int rows = arr.GetLength(0); // y
            int cols = arr.GetLength(1); // x
            int parts = arr.GetLength(2);
            // simgoid to get score
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    for (int part = 0; part < parts; part++)
                    {
                        arr[y, x, part] = Sigmoid(arr[y, x, part]);
                    }
                }
            }
        }


    }
}
