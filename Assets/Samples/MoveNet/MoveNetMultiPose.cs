using UnityEngine;
using UnityEngine.Assertions;

namespace TensorFlowLite
{
    /// <summary>
    /// MoveNet Example
    /// https://www.tensorflow.org/hub/tutorials/movenet
    /// </summary>
    public class MoveNetMultiPose : BaseImagePredictor<int>
    {
        [System.Serializable]
        public class Options
        {
            [FilePopup("*.tflite")]
            public string modelPath = string.Empty;
            public AspectMode aspectMode = AspectMode.Fit;
        }

        // [6, 56]
        // Up to 6 people
        // 17 * 3 (y, x, confidence) + [y_min, x_min, y_max, x_max, score] = 56
        private readonly float[,] outputs0;
        public readonly MoveNetPose[] poses;

        public MoveNetMultiPose(Options options) : base(options.modelPath, true)
        {
            resizeOptions.aspectMode = options.aspectMode;
            int[] outputShape = interpreter.GetOutputTensorInfo(0).shape;

            Assert.AreEqual(1, outputShape[0]);
            Assert.AreEqual(6, outputShape[1]);
            Assert.AreEqual(56, outputShape[2]);

            outputs0 = new float[outputShape[1], outputShape[2]];

            int poseCount = outputShape[1];
            poses = new MoveNetPose[poseCount];
            for (int i = 0; i < poseCount; i++)
            {
                poses[i] = new MoveNetPose();
            }
        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, input0);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
        }

        public MoveNetPose[] GetResults()
        {
            for (int poseIndex = 0; poseIndex < poses.Length; poseIndex++)
            {
                MoveNetPose pose = poses[poseIndex];
                for (int jointIndex = 0; jointIndex < pose.Length; jointIndex++)
                {
                    pose[jointIndex] = new MoveNetPose.Joint(
                        y: outputs0[poseIndex, jointIndex * 3 + 0],
                        x: outputs0[poseIndex, jointIndex * 3 + 1],
                        score: outputs0[poseIndex, jointIndex * 3 + 2]
                    );
                }
            }
            return poses;
        }
    }
}
