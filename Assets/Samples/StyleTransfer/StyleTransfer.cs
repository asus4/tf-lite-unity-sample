using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace TensorFlowLite
{
    public class StyleTransfer : BaseImagePredictor<float>
    {

        float[] styleBottleneck;

        public StyleTransfer(string modelPath, float[] styleBottleneck) : base(modelPath)
        {
            this.styleBottleneck = styleBottleneck;
        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, inputs);

            interpreter.SetInputTensorData(0, inputs);
            interpreter.SetInputTensorData(1, styleBottleneck);

        }

    }
}