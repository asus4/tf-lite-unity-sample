
namespace TensorFlowLite
{
    using UnityEngine;
    using UnityEngine.Assertions;

    public class PoseSegmentation : System.IDisposable
    {
        private readonly ComputeShader compute;
        private readonly ComputeBuffer labelBuffer;
        private readonly RenderTexture maskTex;
        // private RenderTexture 

        private readonly int kLabelToTex;
        private static readonly int kLabelBuffer = Shader.PropertyToID("LabelBuffer");
        private static readonly int kInputTexture = Shader.PropertyToID("InputTexture");
        private static readonly int kOutputTexture = Shader.PropertyToID("OutputTexture");
        private static readonly int kSigmaColor = Shader.PropertyToID("sigmaColor");

        private readonly int width;
        private readonly int height;

        public PoseSegmentation(Interpreter.TensorInfo info, ComputeShader compute)
        {
            this.compute = compute;

            width = info.shape[2];
            height = info.shape[1];
            int channels = info.shape[3];

            Assert.AreEqual(1, channels);

            compute.SetInt("Width", width);
            compute.SetInt("Height", height);

            compute.SetFloat("sigmaTexel", Mathf.Max(1f / width, 1f / height));
            compute.SetInt("step", 1);
            compute.SetInt("radius", 1);

            labelBuffer = new ComputeBuffer(height * width, sizeof(float) * channels);

            maskTex = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
            maskTex.enableRandomWrite = true;
            maskTex.Create();

            kLabelToTex = compute.FindKernel("LabelToTex");
        }

        public void Dispose()
        {
            labelBuffer?.Release();

            if (maskTex != null)
            {
                maskTex.Release();
                Object.Destroy(maskTex);
            }
        }

        public RenderTexture GetTexture(Texture inputTex, float[,] data, float sigmaColor)
        {
            // Label to Texture
            labelBuffer.SetData(data);
            compute.SetFloat(kSigmaColor, sigmaColor);
            compute.SetBuffer(kLabelToTex, kLabelBuffer, labelBuffer);
            compute.SetTexture(kLabelToTex, kOutputTexture, maskTex);
            compute.Dispatch(kLabelToTex, width / 8, height / 8, 1);
            return maskTex;
        }
    }
}
