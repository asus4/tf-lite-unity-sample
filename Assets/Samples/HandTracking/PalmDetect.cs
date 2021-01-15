using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;
using Cysharp.Threading.Tasks;


namespace TensorFlowLite
{

    public class PalmDetect : BaseImagePredictor<float>
    {


        public struct Result
        {
            public float score;
            public Rect rect;
            public Vector2[] keypoints;
        }

        public const int MAX_PALM_NUM = 4;

        // classificators / scores
        private float[] output0 = new float[2944];

        // regressors / points
        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 17 are 7 hand keypoint x and y coordinates: x1,y1,x2,y2,...x7,y7
        private float[,] output1 = new float[2944, 18];
        private List<Result> results = new List<Result>();
        private SsdAnchor[] anchors;

        public PalmDetect(string modelPath) : base(modelPath, true)
        {
            var options = new SsdAnchorsCalculator.Options()
            {
                inputSizeWidth = 256,
                inputSizeHeight = 256,

                minScale = 0.1171875f,
                maxScale = 0.75f,

                anchorOffsetX = 0.5f,
                anchorOffsetY = 0.5f,

                numLayers = 5,
                featureMapWidth = new int[0],
                featureMapHeight = new int[0],
                strides = new int[] { 8, 16, 32, 32, 32 },

                aspectRatios = new float[] { 1.0f },

                reduceBoxesInLowestLayer = false,
                interpolatedScaleAspectRatio = 1.0f,
                fixedAnchorSize = true,
            };

            anchors = SsdAnchorsCalculator.Generate(options);
            UnityEngine.Debug.AssertFormat(anchors.Length == 2944, "Anchors count must be 2944");
        }

        public override void Invoke(Texture inputTex)
        {
            // const float OFFSET = 128f;
            // const float SCALE = 1f / 128f;
            // ToTensor(inputTex, input0, OFFSET, SCALE);
            ToTensor(inputTex, input0);


            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }

        public async UniTask<List<Result>> InvokeAsync(Texture inputTex, CancellationToken cancellationToken)
        {
            await ToTensorAsync(inputTex, input0, cancellationToken);
            await UniTask.SwitchToThreadPool();

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);

            var results = GetResults();

            await UniTask.SwitchToMainThread(cancellationToken);
            return results;
        }

        public List<Result> GetResults(float scoreThreshold = 0.7f, float iouThreshold = 0.3f)
        {
            results.Clear();

            for (int i = 0; i < anchors.Length; i++)
            {
                float score = MathTF.Sigmoid(output0[i]);
                if (score < scoreThreshold)
                {
                    continue;
                }

                SsdAnchor anchor = anchors[i];

                float sx = output1[i, 0];
                float sy = output1[i, 1];
                float w = output1[i, 2];
                float h = output1[i, 3];

                float cx = sx + anchor.x * width;
                float cy = sy + anchor.y * height;

                cx /= (float)width;
                cy /= (float)height;
                w /= (float)width;
                h /= (float)height;

                var keypoints = new Vector2[7];
                for (int j = 0; j < 7; j++)
                {
                    float lx = output1[i, 4 + (2 * j) + 0];
                    float ly = output1[i, 4 + (2 * j) + 1];
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

        private static List<Result> NonMaxSuppression(List<Result> palms, float iou_threshold)
        {
            var filtered = new List<Result>();

            foreach (Result originalPalm in palms.OrderByDescending(o => o.score))
            {
                bool ignore_candidate = false;
                foreach (Result newPalm in filtered)
                {
                    float iou = originalPalm.rect.IntersectionOverUnion(newPalm.rect);
                    if (iou >= iou_threshold)
                    {
                        ignore_candidate = true;
                        break;
                    }
                }

                if (!ignore_candidate)
                {
                    filtered.Add(originalPalm);
                    if (filtered.Count >= MAX_PALM_NUM)
                    {
                        break;
                    }
                }
            }

            return filtered;
        }



    }
}
