namespace TensorFlowLite.MoveNet
{
    using System.Threading;
    using Cysharp.Threading.Tasks;
    using UnityEngine;
    using UnityEngine.Assertions;

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
        public readonly MoveNetPoseWithBoundingBox[] poses;

        public MoveNetMultiPose(Options options) : base(options.modelPath, Accelerator.GPU)
        {
            resizeOptions.aspectMode = options.aspectMode;
            int[] outputShape = interpreter.GetOutputTensorInfo(0).shape;

            Assert.AreEqual(1, outputShape[0]);
            Assert.AreEqual(6, outputShape[1]);
            Assert.AreEqual(56, outputShape[2]);

            outputs0 = new float[outputShape[1], outputShape[2]];

            int poseCount = outputShape[1];
            poses = new MoveNetPoseWithBoundingBox[poseCount];
            for (int i = 0; i < poseCount; i++)
            {
                poses[i] = new MoveNetPoseWithBoundingBox();
            }
        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, inputTensor);

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
        }

        public async UniTask<MoveNetPoseWithBoundingBox[]> InvokeAsync(Texture inputTex, CancellationToken cancellationToken)
        {
            await ToTensorAsync(inputTex, inputTensor, cancellationToken);
            await UniTask.SwitchToThreadPool();

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
            await UniTask.SwitchToMainThread(PlayerLoopTiming.Update, cancellationToken);

            return GetResults();
        }

        public MoveNetPoseWithBoundingBox[] GetResults()
        {
            for (int poseIndex = 0; poseIndex < poses.Length; poseIndex++)
            {
                var pose = poses[poseIndex];
                for (int jointIndex = 0; jointIndex < pose.Length; jointIndex++)
                {
                    pose[jointIndex] = new MoveNetPose.Joint(
                        y: outputs0[poseIndex, jointIndex * 3 + 0],
                        x: outputs0[poseIndex, jointIndex * 3 + 1],
                        score: outputs0[poseIndex, jointIndex * 3 + 2]
                    );

                }

                const int BOX_OFFSET = MoveNetPose.JOINT_COUNT * 3;
                pose.boundingBox = Rect.MinMaxRect(
                    outputs0[poseIndex, BOX_OFFSET + 1],
                    outputs0[poseIndex, BOX_OFFSET + 0],
                    outputs0[poseIndex, BOX_OFFSET + 3],
                    outputs0[poseIndex, BOX_OFFSET + 2]
                );
                pose.score = outputs0[poseIndex, BOX_OFFSET + 4];
            }
            return poses;
        }
    }
}
