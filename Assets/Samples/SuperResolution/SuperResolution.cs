using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

using DataType = TensorFlowLite.Interpreter.DataType;

namespace TensorFlowLite
{
    public sealed class SuperResolution : BaseVisionTask
    {
        private readonly NativeArray<float> output0;
        private readonly TensorToTexture tensorToTexture;

        public RenderTexture ResultTexture => tensorToTexture.OutputTexture;

        public SuperResolution(string modelPath, ComputeShader compute)
        {

            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AddGpuDelegate();
            Load(FileUtil.LoadFile(modelPath), interpreterOptions);

            // Setup output
            var outputInfo = interpreter.GetOutputTensorInfo(0);
            var outputShape = outputInfo.shape;
            int height = outputShape[1];
            int width = outputShape[2];
            int channels = outputShape[3];
            output0 = new NativeArray<float>(outputInfo.GetElementCount(), Allocator.Persistent);

            Debug.Assert(height % 8 == 0);
            Debug.Assert(width % 8 == 0);
            Debug.Assert(channels == 3);

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

            compute.SetInts("_InputSize", new int[] { width, height });
        }

        public override void Dispose()
        {
            tensorToTexture?.Dispose();
            base.Dispose();
        }

        protected override void PostProcess()
        {
            interpreter.GetOutputTensorData(0, output0.AsSpan());
            tensorToTexture.Convert(output0);
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
    }
}
