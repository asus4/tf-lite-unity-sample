using UnityEngine;
using UnityEngine.Assertions;

namespace TensorFlowLite
{
    /// <summary>
    /// MoveNet Example
    /// https://www.tensorflow.org/hub/tutorials/movenet
    /// </summary>
    public class MoveNetSinglePose : BaseImagePredictor<sbyte>
    {
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

        private readonly float[,] outputs0;
        public readonly Result[] results;

        public MoveNetSinglePose(Options options) : base(options.modelPath, true)
        {
            resizeOptions.aspectMode = options.aspectMode;
            var odim0 = interpreter.GetOutputTensorInfo(0).shape;

            Assert.AreEqual(1, odim0[0]);
            Assert.AreEqual(1, odim0[1]);
            Assert.AreEqual(17, odim0[2]);
            Assert.AreEqual(3, odim0[3]);

            outputs0 = new float[odim0[2], odim0[3]];
            results = new Result[odim0[2]];
        }


        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, input0);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
        }

        public Result[] GetResults()
        {
            for (int i = 0; i < results.Length; i++)
            {
                results[i] = new Result(
                    x: outputs0[i, 1],
                    y: outputs0[i, 0],
                    confidence: outputs0[i, 2]
                );
            }
            return results;
        }
    }
}
