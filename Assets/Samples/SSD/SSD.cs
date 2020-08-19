using UnityEngine;

namespace TensorFlowLite
{
    public class SSD : BaseImagePredictor<sbyte>
    {
        public struct Result
        {
            public int classID;
            public float score;
            public Rect rect;
        }


        // https://www.tensorflow.org/lite/models/object_detection/overview

        float[,] outputs0 = new float[10, 4]; // [top, left, bottom, right] * 10
        float[] outputs1 = new float[10]; // Classes
        float[] outputs2 = new float[10]; // Scores
        Result[] results = new Result[10];

        public SSD(string modelPath) : base(modelPath, true)
        {
        }


        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, input0);
            // ToTensor(inputTex, ref input);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
            interpreter.GetOutputTensorData(1, outputs1);
            interpreter.GetOutputTensorData(2, outputs2);
        }

        public Result[] GetResults()
        {
            for (int i = 0; i < 10; i++)
            {
                // Invert Y to adapt Unity UI space
                float top = 1f - outputs0[i, 0];
                float left = outputs0[i, 1];
                float bottom = 1f - outputs0[i, 2];
                float right = outputs0[i, 3];

                results[i] = new Result()
                {
                    classID = (int)outputs1[i],
                    score = outputs2[i],
                    rect = new Rect(left, top, right - left, top - bottom),
                };
            }
            return results;
        }
    }
}
