
using UnityEngine;

namespace TensorFlowLite
{

    public sealed class SelfieSegmentation : BaseImagePredictor<float>
    {
        private float[,,] output0; // height, width, 2

        private readonly ComputeShader compute;
        private ComputeBuffer labelBuffer;
        private RenderTexture labelTex;
        private RenderTexture maskTex;

        private readonly int kLabelToTex;
        private readonly int kBilateralFilter;
        private static readonly int kLabelBuffer = Shader.PropertyToID("LabelBuffer");
        private static readonly int kInputTexture = Shader.PropertyToID("InputTexture");
        private static readonly int kOutputTexture = Shader.PropertyToID("OutputTexture");

        public SelfieSegmentation(string modelPath, ComputeShader compute) : base(modelPath, true)
        {
            int[] odim0 = interpreter.GetOutputTensorInfo(0).shape;

            Debug.Assert(odim0[1] == height);
            Debug.Assert(odim0[2] == width);

            output0 = new float[odim0[1], odim0[2], odim0[3]];

            this.compute = compute;
            compute.SetInt("Width", width);
            compute.SetInt("Height", height);

            labelBuffer = new ComputeBuffer(height * width, sizeof(float) * 2);

            labelTex = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
            labelTex.enableRandomWrite = true;
            labelTex.Create();

            maskTex = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
            maskTex.enableRandomWrite = true;
            maskTex.Create();

            kLabelToTex = compute.FindKernel("LabelToTex");
            kBilateralFilter = compute.FindKernel("BilateralFilter");
        }

        public override void Dispose()
        {
            output0 = null;
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

            base.Dispose();
        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, input0);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
        }

        public RenderTexture GetResultTexture()
        {
            labelBuffer.SetData(output0);
            compute.SetBuffer(kLabelToTex, kLabelBuffer, labelBuffer);
            compute.SetTexture(kLabelToTex, kOutputTexture, labelTex);
            compute.Dispatch(kLabelToTex, width / 8, height / 8, 1);

            compute.SetTexture(kBilateralFilter, kInputTexture, labelTex);
            compute.SetTexture(kBilateralFilter, kOutputTexture, maskTex);
            compute.Dispatch(kBilateralFilter, width / 8, height / 8, 1);
            return maskTex;
        }
    }
}
