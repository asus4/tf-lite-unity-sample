
using UnityEngine;

namespace TensorFlowLite
{

    public sealed class MeetSegmentation : BaseImagePredictor<float>
    {
        float[,,] output0; // height, width, 2

        ComputeShader compute;
        ComputeBuffer labelBuffer;
        RenderTexture labelTex;

        public MeetSegmentation(string modelPath, ComputeShader compute) : base(modelPath, true)
        {
            var odim0 = interpreter.GetOutputTensorInfo(0).shape;

            Debug.Assert(odim0[1] == height);
            Debug.Assert(odim0[2] == width);

            output0 = new float[odim0[1], odim0[2], odim0[3]];

            this.compute = compute;
            compute.SetInt("Width", width);
            compute.SetInt("Height", height);

            labelTex = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
            labelTex.enableRandomWrite = true;
            labelTex.Create();
            labelBuffer = new ComputeBuffer(height * width, sizeof(float) * 2);

            Debug.Log(compute);
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
            compute.SetBuffer(0, "LabelBuffer", labelBuffer);
            compute.SetTexture(0, "Result", labelTex);

            compute.Dispatch(0, width / 8, height / 8, 1);

            return labelTex;
        }
    }
}
