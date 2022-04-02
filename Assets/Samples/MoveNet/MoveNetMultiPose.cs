using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace TensorFlowLite
{
    /// <summary>
    /// MoveNet Example
    /// https://www.tensorflow.org/hub/tutorials/movenet
    /// </summary>
    public class MoveNetMultiPose : BaseImagePredictor<sbyte>
    {
        [System.Serializable]
        public class Options
        {
            [FilePopup("*.tflite")]
            public string modelPath = string.Empty;
            public AspectMode aspectMode = AspectMode.Fit;
        }

        private readonly float[,] outputs0;
        public readonly MoveNetPose pose;

        public MoveNetMultiPose(Options options) : base(options.modelPath, false)
        {
            resizeOptions.aspectMode = options.aspectMode;
            int[] outputShape = interpreter.GetOutputTensorInfo(0).shape;

            Assert.AreEqual(1, outputShape[0]);
            Assert.AreEqual(6, outputShape[1]);
            Assert.AreEqual(56, outputShape[2]);

            outputs0 = new float[outputShape[1], outputShape[2]];
            pose = new MoveNetPose();
        }


        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, input0);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
        }

        public MoveNetPose GetResults()
        {
            return null;
            // for (int i = 0; i < pose.Length; i++)
            // {
            //     pose[i] = new MoveNetPose.Joint(
            //         x: outputs0[i, 1],
            //         y: outputs0[i, 0],
            //         confidence: outputs0[i, 2]
            //     );
            // }
            // return pose;
        }
    }
}
