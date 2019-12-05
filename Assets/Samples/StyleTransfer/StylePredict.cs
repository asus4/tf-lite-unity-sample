using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TensorFlowLite
{
    public class StylePredict : BaseImagePredictor<float>
    {
        float[] output0;

        public StylePredict(string modelPath) : base(modelPath)
        {

            var outDim0 = interpreter.GetOutputTensorInfo(0).shape;
            output0 = new float[outDim0[3]]; // shold be 100

        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, inputs);

            interpreter.SetInputTensorData(0, inputs);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
        }

        public float[] GetStyleBottleneck()
        {
            return output0;
        }
    }
}
