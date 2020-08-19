using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;

namespace TensorFlowLite
{
    public class PoseDetect : BaseImagePredictor<float>
    {
        public struct Result
        {
            public float score;
            public Rect rect;
            public float2x4 keypoints;
        }

        // regressors / points
        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 11 are 4 keypoint x and y coordinates: x0,y0,x1,y1,x2,y2,x3,y3

        // KeyPoint
        // MidHipCenter: 0
        // FullBodySizeRot: 1
        // MidShoulderCenter: 2
        // UpperBodySizeRot: 3
        private float[,] output0 = new float[896, 12];

        // classificators / scores
        private float[] output1 = new float[896];

        public PoseDetect(string modelPath) : base(modelPath, true)
        {

        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, input0);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }

        public Result GetResults()
        {
            Debug.Log("TODO get results");
            return new Result()
            {
                score = 0,
                rect = default(Rect),
                keypoints = default(float2x4),
            };
        }
    }
}
