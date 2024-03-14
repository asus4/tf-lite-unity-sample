using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// TensorFlow Lite Style Transfer Example
    /// https://www.tensorflow.org/lite/examples/style_transfer/overview
    /// </summary>
    public class StyleTransfer : BaseVisionTask<float>
    {
        private readonly float[] styleBottleneck;
        private readonly float[,,] output0;
        private readonly RenderTexture outputTex;
        private readonly ComputeShader compute;
        private readonly ComputeBuffer outputBuffer;

        public StyleTransfer(string modelPath, float[] styleBottleneck, ComputeShader compute)
            : base(FileUtil.LoadFile(modelPath), CreateOptions(TfLiteDelegateType.GPU))
        {
            this.styleBottleneck = styleBottleneck;
            this.compute = compute;

            output0 = new float[height, width, channels];

            outputTex = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat)
            {
                enableRandomWrite = true
            };
            outputTex.Create();

            outputBuffer = new ComputeBuffer(width * height, sizeof(float) * 3);

            AspectMode = AspectMode.Fill;
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

        protected override void PreProcess(Texture texture)
        {
            base.PreProcess(texture);
            interpreter.SetInputTensorData(1, styleBottleneck);
        }

        protected override void PostProcess()
        {
            interpreter.GetOutputTensorData(0, output0);
        }

        public RenderTexture GetResultTexture()
        {
            outputBuffer.SetData(output0);
            compute.SetBuffer(0, "InputTensor", outputBuffer);
            compute.SetTexture(0, "OutputTexture", outputTex);
            compute.Dispatch(0, width / 8, height / 8, 1);
            return outputTex;
        }
    }
}
