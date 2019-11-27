using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;

namespace TensorFlowLite
{
    public class StyleTransfer : BaseImagePredictor<float>
    {

        float[] styleBottleneck;
        float[,,] output0;
        RenderTexture outputTex;
        ComputeShader compute;
        ComputeBuffer outputBuffer;

        public StyleTransfer(string modelPath, float[] styleBottleneck, ComputeShader compute) : base(modelPath, false)
        {
            this.styleBottleneck = styleBottleneck;
            this.compute = compute;

            output0 = new float[height, width, channels];

            outputTex = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat);
            outputTex.enableRandomWrite = true;
            outputTex.Create();

            outputBuffer = new ComputeBuffer(width * height, sizeof(float) * 3);
        }

        public override void Dispose()
        {
            base.Dispose();
            if (outputTex != null)
            {
                outputTex.Release();
                Object.Destroy(outputTex);
            }
            outputBuffer?.Dispose();
        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, inputs);

            interpreter.SetInputTensorData(0, inputs);
            interpreter.SetInputTensorData(1, styleBottleneck);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
        }

        public RenderTexture GetResultTexture()
        {
            // Find min max;
            float3 min, max;
            FindMinMax(output0, out min, out max);

            Debug.Log($"min: {min} max: {max}");

            outputBuffer.SetData(output0);
            compute.SetBuffer(0, "InputTensor", outputBuffer);
            compute.SetTexture(0, "OutputTexture", outputTex);
            compute.SetVector("minValue", new float4(min, 0));
            compute.SetVector("maxValue", new float4(max, 0));

            compute.Dispatch(0, width / 8, height / 8, 1);
            return outputTex;
        }

        static void FindMinMax(float[,,] arr, out float3 min, out float3 max)
        {
            int rows = arr.GetLength(0); // y
            int cols = arr.GetLength(1); // x
            int channels = arr.GetLength(2);
            min = new float3(float.MaxValue, float.MaxValue, float.MaxValue);
            max = new float3(float.MinValue, float.MinValue, float.MinValue);

            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    float r = arr[y, x, 0];
                    float g = arr[y, x, 1];
                    float b = arr[y, x, 2];
                    min.x = Mathf.Min(r, min.x);
                    min.y = Mathf.Min(g, min.y);
                    min.z = Mathf.Min(b, min.z);
                    max.x = Mathf.Max(r, max.x);
                    max.y = Mathf.Max(g, max.y);
                    max.z = Mathf.Max(b, max.z);
                }
            }
        }

    }
}