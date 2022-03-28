
namespace TensorFlowLite
{
    using UnityEngine;
    using UnityEngine.Assertions;

    public class PoseSegmentation : System.IDisposable
    {
        private readonly ComputeShader compute;
        private readonly ComputeBuffer labelBuffer;
        private readonly RenderTexture labelTex;
        private RenderTexture maskTex;
        // private RenderTexture 

        private readonly int kernelLabelToTex;
        private readonly int kernelTransformToCameraMask;


        private static readonly int kLabelBuffer = Shader.PropertyToID("_LabelBuffer");
        private static readonly int kInputTexture = Shader.PropertyToID("_InputTexture");
        private static readonly int kOutputTexture = Shader.PropertyToID("_OutputTexture");
        private static readonly int kSigmaColor = Shader.PropertyToID("_SigmaColor");
        private static readonly int kCropMatrix = Shader.PropertyToID("_CropMatrix");
        private static readonly int kCropWidth = Shader.PropertyToID("_CropWidth");
        private static readonly int kCropHeight = Shader.PropertyToID("_CropHeight");


        private readonly int width;
        private readonly int height;


        public PoseSegmentation(Interpreter.TensorInfo info, ComputeShader compute)
        {
            this.compute = compute;

            width = info.shape[2];
            height = info.shape[1];
            int channels = info.shape[3];

            Assert.AreEqual(1, channels);

            compute.SetInt("_Width", width);
            compute.SetInt("_Height", height);

            compute.SetFloat("_SigmaTexel", Mathf.Max(1f / width, 1f / height));
            compute.SetInt("_Step", 1);
            compute.SetInt("_Radius", 1);

            labelBuffer = new ComputeBuffer(height * width, sizeof(float) * channels);

            labelTex = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32)
            {
                enableRandomWrite = true,
            };
            labelTex.Create();

            kernelLabelToTex = compute.FindKernel("LabelToTex");
            kernelTransformToCameraMask = compute.FindKernel("TransformToCameraMask");
        }

        public void Dispose()
        {
            labelBuffer?.Release();

            TryDestroy(labelTex);
            TryDestroy(maskTex);
        }

        public RenderTexture GetTexture(Texture inputTex, Matrix4x4 cropMatrix, float[,] data, float sigmaColor)
        {
            // Label to Texture with bilateral filter
            labelBuffer.SetData(data);
            compute.SetFloat(kSigmaColor, sigmaColor);
            compute.SetBuffer(kernelLabelToTex, kLabelBuffer, labelBuffer);
            compute.SetTexture(kernelLabelToTex, kOutputTexture, labelTex);
            compute.Dispatch(kernelLabelToTex, width / 8, height / 8, 1);

            // Resize Mask to original texture
            if (maskTex == null || maskTex.width != inputTex.width || maskTex.height != inputTex.height)
            {
                TryDestroy(maskTex);
                maskTex = new RenderTexture(inputTex.width, inputTex.height, 0, RenderTextureFormat.ARGB32)
                {
                    enableRandomWrite = true,
                };
                maskTex.Create();
                compute.SetInt(kCropWidth, inputTex.width);
                compute.SetInt(kCropHeight, inputTex.height);
            }

            compute.SetMatrix(kCropMatrix, cropMatrix);
            compute.SetTexture(kernelTransformToCameraMask, kInputTexture, labelTex);
            compute.SetTexture(kernelTransformToCameraMask, kOutputTexture, maskTex);
            compute.Dispatch(kernelTransformToCameraMask, maskTex.width / 8, maskTex.height / 8, 1);

            return maskTex;
        }

        private static void TryDestroy(RenderTexture tex)
        {
            if (tex != null)
            {
                tex.Release();
                Object.Destroy(tex);
            }
        }

    }
}
