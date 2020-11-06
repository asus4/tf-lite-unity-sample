
using System.Linq;
using UnityEngine;

namespace TensorFlowLite
{

    public sealed class MeetSegmentation : BaseImagePredictor<float>
    {
        float[,,] outputs0; // height, width, 21

        public MeetSegmentation(string modelPath, ComputeShader compute) : base(modelPath, true)
        {
            var odim0 = interpreter.GetOutputTensorInfo(0).shape;

            Debug.Assert(odim0[1] == height);
            Debug.Assert(odim0[2] == width);

            outputs0 = new float[odim0[1], odim0[2], odim0[3]];
        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, input0);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
        }
    }
}
