using UnityEngine;
using UnityEngine.Rendering;

namespace TensorFlowLite
{
    public class SuperResolution : BaseVisionTask
    {
        readonly float[,,] output0;
        readonly ComputeShader compute;
        readonly ComputeBuffer resultBuffer;
        readonly RenderTexture resultTexture;
        readonly int outputWidth, outputHeight;


        public SuperResolution(string modelPath, ComputeShader compute)
        {

            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AutoAddDelegate(TfLiteDelegateType.GPU, typeof(float));
            Load(FileUtil.LoadFile(modelPath), interpreterOptions);

            // Setup output
            var outputShape = interpreter.GetOutputTensorInfo(0).shape;
            outputHeight = outputShape[1];
            outputWidth = outputShape[2];
            int channels = outputShape[3];
            output0 = new float[outputHeight, outputWidth, channels];

            Debug.Assert(outputHeight % 8 == 0);
            Debug.Assert(outputWidth % 8 == 0);
            Debug.Assert(channels == 3);

            // Setup compute
            this.compute = compute;
            compute.SetInt("Width", outputWidth);
            compute.SetInt("Height", outputHeight);

            resultBuffer = new ComputeBuffer(outputWidth * outputHeight, sizeof(float) * channels);
            resultTexture = new RenderTexture(outputWidth, outputHeight, 0, RenderTextureFormat.ARGB32);
            resultTexture.enableRandomWrite = true;
            resultTexture.Create();
        }

        public override void Dispose()
        {
            resultBuffer.Release();
            resultTexture.Release();
            Object.Destroy(resultTexture);

            base.Dispose();
        }

        protected override void PostProcess()
        {
            interpreter.GetOutputTensorData(0, output0);
        }

        protected override TextureToNativeTensor CreateTextureToTensor(Interpreter.TensorInfo inputTensorInfo)
        {
            // ESR-GAN model accepts float but value range must be 0 ~ 255
            var compute = TextureToNativeTensor.CloneDefaultComputeShaderFloat32();
            var keyword = new LocalKeyword(compute, "USE_OFFSET");
            compute.SetKeyword(keyword, true);
            compute.SetFloats("_Mean", new float[] { 0.0f, 0.0f, 0.0f });
            compute.SetFloats("_StdDev", new float[] { 1 / 255f, 1 / 255f, 1 / 255f });

            return TextureToNativeTensor.Create(new()
            {
                compute = compute,
                kernel = 0,
                width = width,
                height = height,
                channels = channels,
                inputType = inputTensorInfo.type,
            });
        }

        public RenderTexture GetResult()
        {
            resultBuffer.SetData(output0);
            compute.SetBuffer(0, "InputBuffer", resultBuffer);
            compute.SetTexture(0, "OutputImage", resultTexture);

            compute.Dispatch(0, outputWidth / 8, outputHeight / 8, 1);

            return resultTexture;
        }
    }
}
