using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TensorFlowLite
{
    public class StylePredict : BaseImagePredictor<float>
    {
        float[] output0;

        public StylePredict(string modelPath) : base(modelPath, true)
        {

            var outDim0 = interpreter.GetOutputTensorInfo(0).shape;
            output0 = new float[outDim0[3]]; // shold be 100
        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, input0);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
        }

        public float[] GetStyleBottleneck()
        {
            return output0;
        }
    }
}
