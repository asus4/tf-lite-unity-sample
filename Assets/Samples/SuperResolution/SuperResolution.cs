using UnityEngine;

namespace TensorFlowLite
{
    public class SuperResolution : BaseImagePredictor<float>
    {
        float[,,] output0;
        ComputeShader compute;
        ComputeBuffer resultBuffer;
        RenderTexture resultTexture;

        int outputWidth, outputHeight;

        public SuperResolution(string modelPath, ComputeShader compute) : base(modelPath, Accelerator.GPU)
        {
            // Setup output
            var odim0 = interpreter.GetOutputTensorInfo(0).shape;
            outputHeight = odim0[1];
            outputWidth = odim0[2];
            int channels = odim0[3];
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

        public override void Invoke(Texture inputTex)
        {
            if (IsConvertSkippable(inputTex))
            {
                ToTensorDirect(inputTex as Texture2D, inputTensor);
            }
            else
            {
                // ToTensor(inputTex, input0, 0f, 255f);
                ToTensor(inputTex, inputTensor);
            }

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
        }

        private void ToTensorDirect(Texture2D tex, float[,,] input)
        {
            var pixels = tex.GetPixels32();
            Debug.Assert(
                pixels.Length == tex.width * tex.height,
                $"length should be {tex.width * tex.height * 3} but was {pixels.Length}");

            // 0 ~ 255
            int w = tex.width;
            int h = tex.height - 1;
            for (int i = 0; i < pixels.Length; i++)
            {
                int y = h - i / w;
                int x = i % w;
                input[y, x, 0] = (float)pixels[i].r;
                input[y, x, 1] = (float)pixels[i].g;
                input[y, x, 2] = (float)pixels[i].b;
            }
        }

        private bool IsConvertSkippable(Texture tex)
        {
            if (!(tex is Texture2D)
            || tex.width != width
            || tex.height != height)
            {
                return false;
            }
            return true;
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
