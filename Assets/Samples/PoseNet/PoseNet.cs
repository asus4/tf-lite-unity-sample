using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

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

        public static readonly Part[,] Connections = new Part[,]
        {
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


        Interpreter interpreter;
        TextureToTensor tex2tensor;
        Result[] results = new Result[17];


        int width;
        int height;
        int channels; // RGB

        float[,,] inputs;
        float[,,] outputs0; // heatmap
        float[,,] outputs1; // offset

        // float[] outputs2 = new float[9 * 9 * 32]; // displacement fwd
        // float[] outputs3 = new float[9 * 9 * 32]; // displacement bwd

        TextureToTensor.ResizeOptions resizeOptions;

        public PoseNet(string modelPath)
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
            interpreter.LogIOInfo();

            var idim0 = interpreter.GetInputTensorInfo(0).dimensions;
            var odim0 = interpreter.GetOutputTensorInfo(0).dimensions;
            var odim1 = interpreter.GetOutputTensorInfo(1).dimensions;

            height = idim0[1];
            width = idim0[2];
            channels = idim0[3];

            inputs = new float[height, width, channels];
            outputs0 = new float[odim0[1], odim0[2], odim0[3]];
            outputs1 = new float[odim1[1], odim1[2], odim1[3]];

            interpreter.ResizeInputTensor(0, idim0);
            interpreter.AllocateTensors();

            tex2tensor = new TextureToTensor();
            resizeOptions = new TextureToTensor.ResizeOptions()
            {
                aspectMode = TextureToTensor.AspectMode.Fill,
                flipX = false,
                flipY = true,
                width = width,
                height = height,
            };
        }


        public void Dispose()
        {
            interpreter?.Dispose();
            tex2tensor?.Dispose();
        }

        public Texture2D inputTex => tex2tensor.texture;

        public void Invoke(Texture inputTex)
        {
            RenderTexture tex = tex2tensor.Resize(inputTex, resizeOptions);
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
            float stride = scores.GetLength(0) - 1;

            ApplySigmoid(scores);
            var argmax = ArgMax2D(scores);

            // Add offsets
            for (int part = 0; part < results.Length; part++)
            {
                ArgMaxResult arg = argmax[part];
                Result res = results[part];

                float offsetX = offsets[arg.y, arg.x, part * 2];
                float offsetY = offsets[arg.y, arg.x, part * 2 + 1];
                res.x = ((float)arg.x / stride * width + offsetX) / width;
                res.y = ((float)arg.y / stride * height + offsetY) / height;
                res.confidence = arg.score;
                res.part = (Part)part;

                results[part] = res;
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

        struct ArgMaxResult
        {
            public int x;
            public int y;
            public float score;
        }

        static ArgMaxResult[] ArgMax2D(float[,,] scores)
        {
            int rows = scores.GetLength(0); //y
            int cols = scores.GetLength(1); //x
            int parts = scores.GetLength(2);

            // Init with minimum float
            var results = new ArgMaxResult[parts];
            for (int i = 0; i < parts; i++)
            {
                results[i].score = float.MinValue;
            }

            // ArgMax
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    for (int part = 0; part < parts; part++)
                    {
                        float current = scores[y, x, part];
                        if (current > results[part].score)
                        {
                            results[part] = new ArgMaxResult()
                            {
                                x = x,
                                y = y,
                                score = current,
                            };
                        }
                    }
                }
            }
            return results;
        }


    }
}
