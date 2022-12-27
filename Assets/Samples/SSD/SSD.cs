using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// Object Detection
    /// See https://www.tensorflow.org/lite/models/object_detection/overview
    /// </summary>
    public class SSD : BaseImagePredictor<sbyte>
    {
        [System.Serializable]
        public class Options
        {
            [FilePopup("*.tflite")]
            public string modelPath = string.Empty;
            public AspectMode aspectMode = AspectMode.Fit;
            public Accelerator accelerator = Accelerator.GPU;
        }

        public readonly struct Result
        {
            public readonly int classID;
            public readonly float score;
            public readonly Rect rect;

            public Result(int classID, float score, Rect rect)
            {
                this.classID = classID;
                this.score = score;
                this.rect = rect;
            }
        }

        const int MAX_DETECTION = 10;
        private readonly float[,] outputs0 = new float[MAX_DETECTION, 4]; // [top, left, bottom, right] * 10
        private readonly float[] outputs1 = new float[MAX_DETECTION]; // Classes
        private readonly float[] outputs2 = new float[MAX_DETECTION]; // Scores
        private readonly Result[] results = new Result[MAX_DETECTION];

        public SSD(Options options)
            : base(options.modelPath, options.accelerator)
        {
            resizeOptions.aspectMode = options.aspectMode;
        }

        public SSD(Options options, InterpreterOptions interpreterOptions)
            : base(options.modelPath, interpreterOptions)
        {
            resizeOptions.aspectMode = options.aspectMode;
        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, inputTensor);
            // ToTensor(inputTex, ref input);

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
            interpreter.GetOutputTensorData(1, outputs1);
            interpreter.GetOutputTensorData(2, outputs2);
        }

        public Result[] GetResults()
        {
            for (int i = 0; i < MAX_DETECTION; i++)
            {
                // Invert Y to adapt Unity UI space
                float top = 1f - outputs0[i, 0];
                float left = outputs0[i, 1];
                float bottom = 1f - outputs0[i, 2];
                float right = outputs0[i, 3];

                results[i] = new Result(
                    classID: (int)outputs1[i],
                    score: outputs2[i],
                    rect: new Rect(left, top, right - left, top - bottom));
            }
            return results;
        }
    }
}
