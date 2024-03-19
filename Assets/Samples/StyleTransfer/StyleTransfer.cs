using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// TensorFlow Lite Style Transfer Example
    /// https://www.tensorflow.org/lite/examples/style_transfer/overview
    /// </summary>
    public sealed class StyleTransfer : BaseVisionTask
    {

        private readonly float[] styleBottleneck;
        private float[,,] output0;

        private readonly ComputeShader compute;
        private TensorToTexture tensorToTexture;

        public RenderTexture ResultTexture => tensorToTexture.OutputTexture;

        public StyleTransfer(string modelPath, float[] styleBottleneck, ComputeShader compute)
        {
            this.compute = compute;

            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AddGpuDelegate();
            Load(FileUtil.LoadFile(modelPath), interpreterOptions);
            AspectMode = AspectMode.Fill;

            this.styleBottleneck = styleBottleneck;
        }

        protected override void InitializeInputsOutputs(Interpreter.TensorInfo inputTensorInfo)
        {
            base.InitializeInputsOutputs(inputTensorInfo);

            var outputInfo = interpreter.GetOutputTensorInfo(0);
            var outputShape = outputInfo.shape;
            int height = outputShape[1];
            int width = outputShape[2];
            int channels = outputShape[3];

            output0 = new float[height, width, channels];
            // Setup compute
            tensorToTexture = new TensorToTexture(new TensorToTexture.Options()
            {
                compute = compute,
                kernel = 0,
                width = width,
                height = height,
                channels = channels,
                inputType = outputInfo.type,
            });

        }

        public override void Dispose()
        {
            tensorToTexture?.Dispose();
            base.Dispose();
        }

        protected override void PreProcess(Texture texture)
        {
            base.PreProcess(texture);
            interpreter.SetInputTensorData(1, styleBottleneck);
        }

        protected override void PostProcess()
        {
            interpreter.GetOutputTensorData(0, output0);
            tensorToTexture.Convert(output0);
        }
    }
}
