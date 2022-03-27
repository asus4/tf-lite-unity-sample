
namespace TensorFlowLite
{
    using UnityEngine;
    using UnityEngine.Assertions;

    public class PoseSegmentation : System.IDisposable
    {
        private readonly ComputeShader compute;
        private ComputeBuffer labelBuffer;
        private RenderTexture labelTex;
        private RenderTexture maskTex;

        private readonly int kLabelToTex;
        private readonly int kBilateralFilter;
        private static readonly int kLabelBuffer = Shader.PropertyToID("LabelBuffer");
        private static readonly int kInputTexture = Shader.PropertyToID("InputTexture");
        private static readonly int kOutputTexture = Shader.PropertyToID("OutputTexture");

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

            labelTex = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
            labelTex.enableRandomWrite = true;
            labelTex.Create();

            maskTex = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
            maskTex.enableRandomWrite = true;
            maskTex.Create();

            kLabelToTex = compute.FindKernel("LabelToTex");
            kBilateralFilter = compute.FindKernel("BilateralFilter");
        }

        public void Dispose()
        {
            if (labelTex != null)
            {
                labelTex.Release();
                Object.Destroy(labelTex);
                labelTex = null;
            }
            if (maskTex != null)
            {
                maskTex.Release();
                Object.Destroy(maskTex);
                maskTex = null;
            }

            labelBuffer?.Release();
            labelBuffer = null;
        }

        public RenderTexture GetTexture(float[,] data, float sigmaColor = 1.0f)
        {
            // Label to Texture
            labelBuffer.SetData(data);
            compute.SetBuffer(kLabelToTex, kLabelBuffer, labelBuffer);
            compute.SetTexture(kLabelToTex, kOutputTexture, labelTex);
            compute.Dispatch(kLabelToTex, width / 8, height / 8, 1);

            // Bilateral Filter
            compute.SetFloat("sigmaColor", sigmaColor);
            compute.SetTexture(kBilateralFilter, kInputTexture, labelTex);
            compute.SetTexture(kBilateralFilter, kOutputTexture, maskTex);
            compute.Dispatch(kBilateralFilter, width / 8, height / 8, 1);
            return maskTex;
        }
    }
}
