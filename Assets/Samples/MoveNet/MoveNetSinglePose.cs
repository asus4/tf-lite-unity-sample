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
    public class MoveNetSinglePose : BaseVisionTask
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

        public MoveNetSinglePose(Options options)
        {
            var interpreterOptions = new InterpreterOptions();
            var delegateType = Application.platform switch
            {
                // Android does not support using GPU delegate with the MoveNet model
                RuntimePlatform.Android => TfLiteDelegateType.NNAPI,
                _ => TfLiteDelegateType.GPU,
            };
            interpreterOptions.AutoAddDelegate(delegateType, typeof(byte));

            Load(FileUtil.LoadFile(options.modelPath), interpreterOptions);

            AspectMode = options.aspectMode;

            int[] outputShape = interpreter.GetOutputTensorInfo(0).shape;

            Assert.AreEqual(1, outputShape[0]);
            Assert.AreEqual(1, outputShape[1]);
            Assert.AreEqual(MoveNetPose.JOINT_COUNT, outputShape[2]);
            Assert.AreEqual(3, outputShape[3]);

            outputs0 = new float[outputShape[2], outputShape[3]];
            pose = new MoveNetPose();
        }

        protected override void PostProcess()
        {
            interpreter.GetOutputTensorData(0, outputs0);
            GetResult();
        }

#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously

        protected override async UniTask PostProcessAsync(CancellationToken cancellationToken)
        {
            interpreter.GetOutputTensorData(0, outputs0);
            GetResult();
        }
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously

        public MoveNetPose GetResult()
        {
            for (int i = 0; i < pose.Length; i++)
            {
                pose[i] = new MoveNetPose.Joint(
                    x: outputs0[i, 1],
                    y: outputs0[i, 0],
                    score: outputs0[i, 2]
                );
            }
            return pose;
        }
    }
}
