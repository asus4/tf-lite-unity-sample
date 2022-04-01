using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using UnityEngine.Assertions;
using Cysharp.Threading.Tasks;

namespace TensorFlowLite
{
    /// <summary>
    /// MoveNet Example
    /// https://www.tensorflow.org/hub/tutorials/movenet
    /// </summary>
    public class MoveNetMultiple : BaseImagePredictor<sbyte>
    {
        public int MultiPoseInstanceSize = 17 * 3 + 5;
        public int numInstance = 0;
        [System.Serializable]
        public class Options
        {
            [FilePopup("*.tflite")]
            public string modelPath = string.Empty;
            public AspectMode aspectMode = AspectMode.Fit;
        }

        public enum Part
        {
            NOSE = 0,
            LEFT_EYE = 1,
            RIGHT_EYE = 2,
            LEFT_EAR = 3,
            RIGHT_EAR = 4,
            LEFT_SHOULDER = 5,
            RIGHT_SHOULDER = 6,
            LEFT_ELBOW = 7,
            RIGHT_ELBOW = 8,
            LEFT_WRIST = 9,
            RIGHT_WRIST = 10,
            LEFT_HIP = 11,
            RIGHT_HIP = 12,
            LEFT_KNEE = 13,
            RIGHT_KNEE = 14,
            LEFT_ANKLE = 15,
            RIGHT_ANKLE = 16,
        }

        public static readonly Part[,] Connections = new Part[,]
        {
            // HEAD
            { Part.LEFT_EAR, Part.LEFT_EYE },
            { Part.LEFT_EYE, Part.NOSE },
            { Part.NOSE, Part.RIGHT_EYE },
            { Part.RIGHT_EYE, Part.RIGHT_EAR },
            // BODY
            { Part.LEFT_HIP, Part.LEFT_SHOULDER },
            { Part.LEFT_ELBOW, Part.LEFT_SHOULDER },
            { Part.LEFT_ELBOW, Part.LEFT_WRIST },
            { Part.LEFT_HIP, Part.LEFT_KNEE },
            { Part.LEFT_KNEE, Part.LEFT_ANKLE },
            { Part.RIGHT_HIP, Part.RIGHT_SHOULDER },
            { Part.RIGHT_ELBOW, Part.RIGHT_SHOULDER },
            { Part.RIGHT_ELBOW, Part.RIGHT_WRIST },
            { Part.RIGHT_HIP, Part.RIGHT_KNEE },
            { Part.RIGHT_KNEE, Part.RIGHT_ANKLE },
            { Part.LEFT_SHOULDER, Part.RIGHT_SHOULDER },
            { Part.LEFT_HIP, Part.RIGHT_HIP }
        };

        [System.Serializable]
        public readonly struct Result
        {
            public readonly float x;
            public readonly float y;
            public readonly float confidence;

            public Result(float x, float y, float confidence)
            {
                this.x = x;
                this.y = y;
                this.confidence = confidence;
            }
        }

        public readonly struct Pose
        {
            public readonly Result[] keypoints;
            public readonly float score;

            public Pose(Result[] keypoints, float score)
            {
                this.score = score;
                this.keypoints = keypoints;
            }

        }

        private readonly float[,] outputs0;
        public readonly Result[] results;

        public Pose[] poses;

        public MoveNetMultiple(Options options) : base(options.modelPath, false)
        {
            resizeOptions.aspectMode = options.aspectMode;
            // Debug.Log(interpreter);
            var odim0 = interpreter.GetOutputTensorInfo(0).shape;

            Debug.Log($"output0: {odim0[0]}, {odim0[1]}, {odim0[2]}");
            Assert.AreEqual(odim0.Length, 3);
            Assert.AreEqual(odim0[0], 1);
            Assert.AreEqual(odim0[2], MultiPoseInstanceSize);
            numInstance = odim0[1];

            outputs0 = new float[odim0[1], odim0[2]];
            poses = new Pose[odim0[1]];
        }


        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, input0);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
        }


        public Pose[] GetPoses()
        {
            // var numInstances = poses.Length / MultiPoseInstanceSize;
            for (int i = 0; i < numInstance; i++)
            {
                // var scoreIndex = i * MULTIPOSE_INSTANCE_SIZE + MULTIPOSE_BOX_SCORE_IDX;
                for (int j = 0; j < MultiPoseInstanceSize; j++)
                {
                    if (i == 0)
                    {

                        Debug.Log($"{i}, {j}: {outputs0[i, j]}");
                    }
                }
                poses[i] = new Pose(new Result[17], outputs0[i, 55]);
                Debug.Log($"{i}: {outputs0[i, 17 * 3 + 4]}");
                for (int j = 0; j < 17; j++)
                {
                    poses[i].keypoints[j] = new Result(
                        x: outputs0[i, j * 3 + 1],
                        y: outputs0[i, j * 3 + 0],
                        confidence: outputs0[i, j * 3 + 2]
                    );
                }
            }
            return poses;

        }
    }
}
