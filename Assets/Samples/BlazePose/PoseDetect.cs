using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Mathematics;

namespace TensorFlowLite
{
    public class PoseDetect : BaseImagePredictor<float>
    {
        public enum KeyPoint
        {
            MidHipCenter = 0,
            FullBodySizeRot = 1,
            MidShoulderCenter = 2,
            UpperBodySizeRot = 3,
        }

        public struct Result
        {
            public float score;
            public Rect rect;
            public float2x4 keypoints;

            public static Result Negative => new Result() { score = -1, };

            public Vector2 HipCenter => keypoints[(int)KeyPoint.MidHipCenter];
            public Vector2 MidShoulderCenter => keypoints[(int)KeyPoint.MidShoulderCenter];
        }

        const int MAX_POSE_NUM = 100;

        // regressors / points
        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 11 are 4 keypoint x and y coordinates: x0,y0,x1,y1,x2,y2,x3,y3
        private float[,] output0 = new float[896, 12];

        // classificators / scores
        private float[] output1 = new float[896];

        private SsdAnchor[] anchors;
        private List<Result> results = new List<Result>();

        public PoseDetect(string modelPath) : base(modelPath, true)
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

        public Result GetResults(float scoreThreshold = 0.5f, float iouThreshold = 0.3f)
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

                var keypoints = new float2[4];
                for (int j = 0; j < 4; j++)
                {
                    float lx = output0[i, 4 + (2 * j) + 0];
                    float ly = output0[i, 4 + (2 * j) + 1];
                    lx += anchor.x * width;
                    ly += anchor.y * height;
                    lx /= (float)width;
                    ly /= (float)height;
                    keypoints[j] = new float2(lx, ly);
                }

                results.Add(new Result()
                {
                    score = score,
                    rect = new Rect(cx - w * 0.5f, cy - h * 0.5f, w, h),
                    keypoints = new float2x4(keypoints[0], keypoints[1], keypoints[2], keypoints[3]),
                });
            }

            // No result
            if (results.Count == 0)
            {
                return Result.Negative;
            }

            // return results.OrderByDescending(o => o.score).First();
            return NonMaxSuppression(results, iouThreshold).First();
        }

        private static List<Result> NonMaxSuppression(List<Result> results, float iouThreshold)
        {
            var filtered = new List<Result>();

            // FIXME LinQ allocs GC each frame
            foreach (Result original in results.OrderByDescending(o => o.score))
            {
                bool ignoreCandidate = false;
                foreach (Result newResult in filtered)
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
                    filtered.Add(original);
                    if (filtered.Count >= MAX_POSE_NUM)
                    {
                        break;
                    }
                }
            }

            return filtered;
        }
    }
}
