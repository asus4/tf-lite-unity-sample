using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TensorFlowLite
{


    public class FaceDetect : BaseImagePredictor<float>
    {
        public class Result
        {
            public float score;
            public Rect rect;
            public Vector2[] keypoints;
        }

        // regressors / points
        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 11 are 4 keypoint x and y coordinates: x0,y0,x1,y1,x2,y2,x3,y3
        private float[,] output0 = new float[896, 16];

        // classificators / scores
        private float[] output1 = new float[896];

        private SsdAnchor[] anchors;
        private List<Result> results = new List<Result>();

        public FaceDetect(string modelPath) : base(modelPath, true)
        {
            var options = new SsdAnchorsCalcurator.Options()
            {
                inputSizeWidth = 128,
                inputSizeHeight = 128,

                minScale = 0.1484375f,
                maxScale = 0.75f,

                anchorOffsetX = 0.5f,
                anchorOffsetY = 0.5f,

                numLayers = 4,
                featureMapWidth = new int[0],
                featureMapHeight = new int[0],
                strides = new int[] { 8, 16, 16, 16 },

                aspectRatios = new float[] { 1.0f },

                reduceBoxesInLowestLayer = false,
                interpolatedScaleAspectRatio = 1.0f,
                fixedAnchorSize = true,
            };

            anchors = SsdAnchorsCalcurator.Generate(options);
            UnityEngine.Debug.AssertFormat(anchors.Length == 896, $"Anchors count must be 896, but was {anchors.Length}");
        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, input0);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }

        public List<Result> GetResults(float scoreThreshold = 0.7f, float iouThreshold = 0.3f)
        {
            return results;
        }
    }
}
