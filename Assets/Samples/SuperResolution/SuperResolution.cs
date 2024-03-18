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
        readonly Vector2Int outputSize;


        public SuperResolution(string modelPath, ComputeShader compute)
        {

            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AutoAddDelegate(TfLiteDelegateType.GPU, typeof(float));
            Load(FileUtil.LoadFile(modelPath), interpreterOptions);

            // Setup output
            var outputShape = interpreter.GetOutputTensorInfo(0).shape;
            outputSize = new Vector2Int(outputShape[2], outputShape[1]);
            int channels = outputShape[3];
            output0 = new float[outputSize.y, outputSize.x, channels];

            Debug.Assert(outputSize.y % 8 == 0);
            Debug.Assert(outputSize.x % 8 == 0);
            Debug.Assert(channels == 3);

            // Setup compute
            this.compute = compute;
            compute.SetInts("_InputSize", new int[] { outputSize.x, outputSize.y });

            resultBuffer = new ComputeBuffer(outputSize.x * outputSize.y, sizeof(float) * channels);
            resultTexture = new RenderTexture(outputSize.x, outputSize.y, 0, RenderTextureFormat.ARGB32)
            {
                enableRandomWrite = true
            };
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
            compute.SetBuffer(0, "_InputTensor", resultBuffer);
            compute.SetTexture(0, "_OutputTex", resultTexture);

            compute.Dispatch(0, outputSize.x / 8, outputSize.y / 8, 1);

            return resultTexture;
        }
    }
}
