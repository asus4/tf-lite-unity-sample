using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;


namespace TensorFlowLite
{
    /// <summary>
    /// Utility to convert Texture to Tensor
    /// </summary>
    public class TextureToTensor : System.IDisposable
    {

        Texture2D fetchTexture;
        ComputeShader compute;
        ComputeBuffer tensorBuffer;

        public Texture2D texture => fetchTexture;

        static readonly int InputTexture = Shader.PropertyToID("InputTexture");
        static readonly int OutputFloatTensor = Shader.PropertyToID("OutputFloatTensor");
        static readonly int TextureWidth = Shader.PropertyToID("TextureWidth");
        static readonly int TextureHeight = Shader.PropertyToID("TextureHeight");


        public TextureToTensor()
        {
            compute = Resources.Load<ComputeShader>("TextureToTensor");
        }

        public void Dispose()
        {
            DisposeUtil.TryDispose(fetchTexture);
            DisposeUtil.TryDispose(tensorBuffer);
        }

        public void ToTensor(RenderTexture texture, sbyte[,,] inputs)
        {
            // var pixels = FetchToTexture2D(texture).GetPixels32();
            var pixels = FetchToTexture2D(texture).GetRawTextureData<sbyte>();
            int width = texture.width;
            int height = texture.height;

            for (int i = 0; i < width * height; i++)
            {
                int y = i / width;
                int x = i % width;
                inputs[y, x, 0] = (sbyte)pixels[i * 3 + 0];
                inputs[y, x, 1] = (sbyte)pixels[i * 3 + 1];
                inputs[y, x, 2] = (sbyte)pixels[i * 3 + 2];
            }
        }

        public void ToTensor<T>(RenderTexture texture, ref NativeArray<T> inputs) where T : struct
        {
            var pixels = FetchToTexture2D(texture).GetRawTextureData<T>();
            // Debug.Log($"pixels:{pixels.Length} inputs:{inputs.Length}");
            if (pixels.Length != inputs.Length)
            {
                pixels = pixels.GetSubArray(0, inputs.Length);
            }
            inputs.CopyFrom(pixels);
        }

        public void ToTensor(RenderTexture texture, float[,,] inputs)
        {
            if (texture.width % 8 != 0 || texture.height % 8 != 0)
            {
                ToTensorCPU(texture, inputs);
            }
            else
            {
                ToTensorGPU(texture, inputs);
            }
        }


        public void ToTensor(RenderTexture texture, float[,,] inputs, float offset, float scale)
        {
            // TODO: optimize this
            var pixels = FetchToTexture2D(texture).GetPixels32();
            int width = texture.width;
            for (int i = 0; i < pixels.Length; i++)
            {
                int y = i / width;
                int x = i % width;
                inputs[y, x, 0] = (pixels[i].r - offset) * scale;
                inputs[y, x, 1] = (pixels[i].g - offset) * scale;
                inputs[y, x, 2] = (pixels[i].b - offset) * scale;
            }
        }

        void ToTensorCPU(RenderTexture texture, float[,,] inputs)
        {
            var pixels = FetchToTexture2D(texture).GetPixels32();
            int width = texture.width;
            const float scale = 255f;
            for (int i = 0; i < pixels.Length; i++)
            {
                int y = i / width;
                int x = i % width;
                inputs[y, x, 0] = (float)(pixels[i].r) / scale;
                inputs[y, x, 1] = (float)(pixels[i].g) / scale;
                inputs[y, x, 2] = (float)(pixels[i].b) / scale;
            }
        }

        void ToTensorGPU(RenderTexture texture, float[,,] inputs)
        {
            int width = texture.width;
            int height = texture.height;

            if (tensorBuffer == null || tensorBuffer.count != width * height)
            {
                DisposeUtil.TryDispose(tensorBuffer);
                tensorBuffer = new ComputeBuffer(width * height, sizeof(float) * 3);
            }
            int kernel = compute.FindKernel("TextureToFloatTensor");

            compute.SetTexture(kernel, InputTexture, texture);
            compute.SetBuffer(kernel, OutputFloatTensor, tensorBuffer);
            compute.SetInt(TextureWidth, width);
            compute.SetInt(TextureHeight, height);

            compute.Dispatch(kernel, width / 8, height / 8, 1);

            tensorBuffer.GetData(inputs);
        }

        private Texture2D FetchToTexture2D(RenderTexture texture)
        {
            if (fetchTexture == null || !IsSameSize(fetchTexture, texture))
            {
                fetchTexture = new Texture2D(texture.width, texture.height, TextureFormat.RGB24, 0, false);
            }
            var prevRT = RenderTexture.active;
            RenderTexture.active = texture;

            fetchTexture.ReadPixels(new Rect(0, 0, texture.width, texture.height), 0, 0);
            fetchTexture.Apply();

            RenderTexture.active = prevRT;

            return fetchTexture;
        }

        private static bool IsSameSize(Texture a, Texture b)
        {
            return a.width == b.width && a.height == b.height;
        }


    }
}
