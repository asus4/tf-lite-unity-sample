using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace TensorFlowLite
{


    public class FaceDetect : BaseImagePredictor<float>
    {
        public enum KeyPoint
        {
            RightEye,  //  0
            LeftEye, //  1
            Nose, //  2
            Mouth, //  3
            RightEar, //  4
            LeftEar, //  5
        }

        public class Result
        {
            public float score;
            public Rect rect;
            public Vector2[] keypoints;

            public Vector2 rightEye => keypoints[(int)KeyPoint.RightEye];
            public Vector2 leftEye => keypoints[(int)KeyPoint.LeftEye];
        }

        private const int KEY_POINT_SIZE = 6;

        private const int MAX_FACE_NUM = 100;

        // regressors / points
        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 15 are 6 keypoint x and y coordinates: x0,y0,x1,y1,x2,y2,x3,y3
        private float[,] output0 = new float[896, 16];

        // classificators / scores
        private float[] output1 = new float[896];

        private SsdAnchor[] anchors;
        private List<Result> results = new List<Result>();
        private List<Result> filterdResults = new List<Result>();

        public FaceDetect(string modelPath) : base(modelPath, true)
        {
            var options = new SsdAnchorsCalculator.Options()
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

            anchors = SsdAnchorsCalculator.Generate(options);
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
            results.Clear();

            for (int i = 0; i < anchors.Length; i++)
            {
                float score = MathTF.Sigmoid(output1[i]);
                if (score < scoreThreshold)
                {
                    continue;
                }
                SsdAnchor anchor = anchors[i];

                float sx = output0[i, 0];
                float sy = output0[i, 1];
                float w = output0[i, 2];
                float h = output0[i, 3];

                float cx = sx + anchor.x * width;
                float cy = sy + anchor.y * height;

                cx /= (float)width;
                cy /= (float)height;
                w /= (float)width;
                h /= (float)height;

                var keypoints = new Vector2[KEY_POINT_SIZE];
                for (int j = 0; j < KEY_POINT_SIZE; j++)
                {
                    float lx = output0[i, 4 + (2 * j) + 0];
                    float ly = output0[i, 4 + (2 * j) + 1];
                    lx += anchor.x * width;
                    ly += anchor.y * height;
                    lx /= (float)width;
                    ly /= (float)height;
                    keypoints[j] = new Vector2(lx, ly);
                }
                results.Add(new Result()
                {
                    score = score,
                    rect = new Rect(cx - w * 0.5f, cy - h * 0.5f, w, h),
                    keypoints = keypoints,
                });
            }

            return NonMaxSuppression(results, iouThreshold);
        }

        private List<Result> NonMaxSuppression(List<Result> results, float iouThreshold)
        {
            filterdResults.Clear();
            // FIXME LinQ allocs GC each frame
            // Use sorted list
            foreach (Result original in results.OrderByDescending(o => o.score))
            {
                bool ignoreCandidate = false;
                foreach (Result newResult in filterdResults)
                {
                    float iou = original.rect.IntersectionOverUnion(newResult.rect);
                    if (iou >= iouThreshold)
                    {
                        ignoreCandidate = true;
                        break;
                    }
                }

                if (!ignoreCandidate)
                {
                    filterdResults.Add(original);
                    if (filterdResults.Count >= MAX_FACE_NUM)
                    {
                        break;
                    }
                }
            }
            return filterdResults;
        }
    }
}
