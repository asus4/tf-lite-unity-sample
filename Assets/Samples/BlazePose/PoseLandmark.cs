using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TensorFlowLite
{

    public class PoseLandmark : BaseImagePredictor<float>
    {
        public PoseLandmark(string modelPath) : base(modelPath, true)
        {
        }

        public override void Invoke(Texture inputTex)
        {
        }
    }
}
