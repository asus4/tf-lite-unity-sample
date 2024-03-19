using System;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Assertions;

using DataType = TensorFlowLite.Interpreter.DataType;
using Object = UnityEngine.Object;

namespace TensorFlowLite
{
    /// <summary>
    /// Converts tensor to texture
    /// </summary>
    public sealed class TensorToTexture : IDisposable
    {
        [Serializable]
        public class Options
        {
            public ComputeShader compute = null;
            public int kernel = 0;
            public int width = 0;
            public int height = 0;
            public int channels = 0;
            public DataType inputType = DataType.Float32;
        }

        private static readonly int _InputTensor = Shader.PropertyToID("_InputTensor");
        private static readonly int _InputSize = Shader.PropertyToID("_InputSize");
        private static readonly int _OutputTexture = Shader.PropertyToID("_OutputTexture");

        private static readonly Lazy<ComputeShader> DefaultComputeShaderFloat32 = new(()
            => Resources.Load<ComputeShader>("com.github.asus4.tflite.common/TensorToTextureFloat32"));

        private readonly ComputeShader compute;
        private readonly int kernel;
        private readonly int width;
        private readonly int height;
        private readonly int channels;


        private readonly GraphicsBuffer tensorBuffer;
        private readonly RenderTexture outputTexture;

        public RenderTexture OutputTexture => outputTexture;

        public TensorToTexture(Options options)
        {
            compute = options.compute != null
                ? options.compute
                : DefaultComputeShaderFloat32.Value;
            kernel = options.kernel;
            width = options.width;
            height = options.height;
            channels = options.channels;

            Assert.IsNotNull(compute, "ComputeShader is not set");

            int stride = channels * DataTypeToStride(options.inputType);
            tensorBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, width * height, stride);
            outputTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32)
            {
                enableRandomWrite = true
            };
            outputTexture.Create();
        }

        public void Dispose()
        {
            tensorBuffer.Dispose();
            outputTexture.Release();
            Object.Destroy(outputTexture);
        }

        public RenderTexture Convert(Array data)
        {
            tensorBuffer.SetData(data);
            compute.SetInts(_InputSize, width, height);
            compute.SetBuffer(kernel, _InputTensor, tensorBuffer);
            compute.SetTexture(kernel, _OutputTexture, outputTexture);
            compute.Dispatch(kernel, Mathf.CeilToInt(width / 8f), Mathf.CeilToInt(height / 8f), 1);
            return outputTexture;
        }

        public RenderTexture Convert<T>(NativeArray<T> data)
            where T : struct
        {
            tensorBuffer.SetData(data);
            compute.SetInts(_InputSize, width, height);
            compute.SetBuffer(kernel, _InputTensor, tensorBuffer);
            compute.SetTexture(kernel, _OutputTexture, outputTexture);
            compute.Dispatch(kernel, Mathf.CeilToInt(width / 8f), Mathf.CeilToInt(height / 8f), 1);
            return outputTexture;
        }

        private static int DataTypeToStride(DataType type)
        {
            return type switch
            {
                DataType.Float32 => sizeof(float),
                _ => throw new NotSupportedException($"Unsupported type: {type}"),
            };
        }
    }
}
