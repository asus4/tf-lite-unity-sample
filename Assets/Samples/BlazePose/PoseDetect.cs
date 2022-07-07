using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;
using UnityEngine.Assertions;

namespace TensorFlowLite
{
    public sealed class PoseDetect : BaseImagePredictor<float>
    {
        [System.Serializable]
        public class Options
        {
            [FilePopup("*.tflite")]
            public string modelPath = string.Empty;
            public AspectMode aspectMode = AspectMode.Fit;

            [Range(0, 1)]
            public float scoreThreshold = 0.5f;
            public bool useNonMaxSuppression = false;
            [Range(0, 1)]
            public float iouThreshold = 0.3f;
        }

        public class Result : System.IComparable<Result>
        {
            public float score;
            public Rect rect;
            public Vector2[] keypoints;

            public static Result Negative => new Result() { score = -1, };

            public int CompareTo(Result other)
            {
                return score > other.score ? -1 : 1;
            }
        }

        private const int MAX_POSE_NUM = 100;
        private const int ANCHOR_LENGTH = 2254;
        private readonly int keypointsCount;

        // regressors / points
        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 11 are 4 keypoints x and y coordinates: x0,y0,x1,y1,x2,y2,x3,y3
        private readonly float[,] output0 = new float[ANCHOR_LENGTH, 12];

        // classificators / scores
        private readonly float[] output1 = new float[ANCHOR_LENGTH];

        private readonly SsdAnchor[] anchors;
        private readonly SortedSet<Result> results = new SortedSet<Result>();

        private readonly Options options;

        public PoseDetect(Options options) : base(options.modelPath, Accelerator.GPU)
        {
            this.options = options;
            resizeOptions.aspectMode = options.aspectMode;

            var anchorOptions = new SsdAnchorsCalculator.Options()
            {
                inputSizeWidth = width,
                inputSizeHeight = height,

                minScale = 0.1484375f,
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

            anchors = SsdAnchorsCalculator.Generate(anchorOptions);
            Assert.AreEqual(anchors.Length, ANCHOR_LENGTH,
                $"Anchors count must be {ANCHOR_LENGTH}, but was {anchors.Length}");

            // Get Keypoint Mode
            var odim0 = interpreter.GetOutputTensorInfo(0).shape;
            keypointsCount = (odim0[2] - 4) / 2;
        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, inputTensor);

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }

        public async UniTask<Result> InvokeAsync(Texture inputTex, CancellationToken cancellationToken, PlayerLoopTiming timing = PlayerLoopTiming.Update)
        {
            await ToTensorAsync(inputTex, inputTensor, cancellationToken);
            await UniTask.SwitchToThreadPool();

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);

            var results = GetResults();

            await UniTask.SwitchToMainThread(timing, cancellationToken);
            return results;
        }

        public Result GetResults()
        {
            results.Clear();

            for (int i = 0; i < anchors.Length; i++)
            {
                float score = MathTF.Sigmoid(output1[i]);
                if (score < options.scoreThreshold)
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

                cx /= width;
                cy /= height;
                w /= width;
                h /= height;

                var keypoints = new Vector2[keypointsCount];
                for (int j = 0; j < keypointsCount; j++)
                {
                    float lx = output0[i, 4 + (2 * j) + 0];
                    float ly = output0[i, 4 + (2 * j) + 1];
                    lx += anchor.x * width;
                    ly += anchor.y * height;
                    lx /= width;
                    ly /= height;
                    keypoints[j] = new Vector2(lx, ly);
                }

                results.Add(new Result()
                {
                    score = score,
                    rect = new Rect(cx - w * 0.5f, cy - h * 0.5f, w, h),
                    keypoints = keypoints,
                });
            }

            // No result
            if (results.Count == 0)
            {
                return Result.Negative;
            }

            if (options.useNonMaxSuppression)
            {
                return NonMaxSuppression(results, options.iouThreshold).First();
            }
            else
            {
                return results.First();
            }
        }

        private static readonly List<Result> nonMaxSupressionCache = new List<Result>();
        private static List<Result> NonMaxSuppression(SortedSet<Result> results, float iouThreshold)
        {
            nonMaxSupressionCache.Clear();
            foreach (Result original in results)
            {
                bool ignoreCandidate = false;
                foreach (Result newResult in nonMaxSupressionCache)
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
                    nonMaxSupressionCache.Add(original);
                    if (nonMaxSupressionCache.Count >= MAX_POSE_NUM)
                    {
                        break;
                    }
                }
            }

            return nonMaxSupressionCache;
        }
    }
}
