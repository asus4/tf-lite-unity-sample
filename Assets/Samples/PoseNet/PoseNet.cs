using System.Collections;
using System.Collections.Generic;
using System.Linq;
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
        public enum BodyPart
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

        [System.Serializable]
        public struct Result
        {
            public BodyPart part;
            public float confidence;
            public float x;
            public float y;
        }

        const int WIDTH = 257;
        const int HEIGHT = 257;
        const int CHANNELS = 3; // RGB


        Interpreter interpreter;
        RenderTexture resizeTexture;
        Material resizeMat;
        Texture2D fetchTexture;
        Result[] results = new Result[17];

        float[] inputs = new float[WIDTH * HEIGHT * CHANNELS];
        float[] outputs0 = new float[9 * 9 * 17]; // heatmap
        float[] outputs1 = new float[9 * 9 * 34]; // offset
        float[] outputs2 = new float[9 * 9 * 32]; // displacement fwd
        float[] outputs3 = new float[9 * 9 * 32]; // displacement bwd

        public float[] heatmap => outputs0;
        public float[] offsets => outputs1;

        public PoseNet(string modelPath)
        {
            interpreter = new Interpreter(File.ReadAllBytes(modelPath), 2);
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
        }

        public void Dispose()
        {
            interpreter?.Dispose();

            if (resizeTexture != null)
            {
                Object.Destroy(resizeTexture);
                Object.Destroy(resizeMat);
            }
        }

        public void Invoke(Texture inputTex)
        {
            RenderTexture tex = ResizeTexture(inputTex);
            TextureToTensor(tex, inputs);

            interpreter.SetInputTensorData(0, inputs);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
            interpreter.GetOutputTensorData(1, outputs1);
            interpreter.GetOutputTensorData(2, outputs2);
            interpreter.GetOutputTensorData(3, outputs3);
        }

        public Result[] GetResults()
        {
            // simgoid to get score
            // output0 -> scores;
            for (int i = 0; i < outputs0.Length; i++)
            {
                outputs0[i] = Sigmoid(outputs0[i]);
            }
            float[] scores = outputs0;
            float[] offsets = outputs1;

            // argmax2d
            // x, y, score
            const int KEYPOINTS = 17;
            const int ROWS = 9; //y
            const int COLS = 9; //x

            // Reset Keypoints
            Vector3[] posisions = new Vector3[KEYPOINTS];
            for (int i = 0; i < posisions.Length; i++)
            {
                posisions[i] = new Vector3(-1, -1, float.MinValue);
            }

            for (int y = 0; y < ROWS; y++)
            {
                for (int x = 0; x < COLS; x++)
                {
                    for (int key = 0; key < KEYPOINTS; key++)
                    {
                        int index = y * COLS + x;
                        float score = scores[index * KEYPOINTS + key];
                        if (posisions[key].z > score)
                        {
                            posisions[key] = new Vector3(x, y, score);
                        }
                    }
                }
            }

            const int STRIDE = 9 - 1;
            for (int i = 0; i < results.Length; i++)
            {
                int x = (int)posisions[i].x;
                int y = (int)posisions[i].y;

                results[i] = new Result()
                {
                    part = (BodyPart)i,
                    x = x / STRIDE * WIDTH,
                    y = y / STRIDE * HEIGHT,
                    confidence = (int)posisions[i].z,
                };
            }

            return results;
        }

        void TextureToTensor(RenderTexture texture, float[] inputs)
        {
            if (fetchTexture == null)
            {
                fetchTexture = new Texture2D(WIDTH, HEIGHT, TextureFormat.RGB24, 0, false);
            }

            var prevRT = RenderTexture.active;
            RenderTexture.active = texture;

            fetchTexture.ReadPixels(new Rect(0, 0, WIDTH, HEIGHT), 0, 0);
            fetchTexture.Apply();

            RenderTexture.active = prevRT;

            const float offset = 128f;
            var pixels = fetchTexture.GetPixels32();
            for (int i = 0; i < pixels.Length; i++)
            {
                inputs[i * 3] = ((float)pixels[i].r - offset) / offset;
                inputs[i * 3 + 1] = ((float)pixels[i].g - offset) / offset;
                inputs[i * 3 + 2] = ((float)pixels[i].b - offset) / offset;
            }
        }


        RenderTexture ResizeTexture(Texture texture)
        {
            if (resizeTexture == null)
            {
                resizeTexture = new RenderTexture(WIDTH, HEIGHT, 0, RenderTextureFormat.ARGB32);
                resizeMat = new Material(Shader.Find("Hidden/YFlip"));

                resizeMat.SetInt("_FlipX", 0);
                resizeMat.SetInt("_FlipY", 1);
            }
            Graphics.Blit(texture, resizeTexture, resizeMat, 0);
            return resizeTexture;
        }


        static float Sigmoid(float x)
        {
            return (1.0f / (1.0f + Mathf.Exp(-x)));
        }


    }
}
