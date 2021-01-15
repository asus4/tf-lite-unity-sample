using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;
using Cysharp.Threading.Tasks;

namespace TensorFlowLite
{
    public abstract class PoseDetect : BaseImagePredictor<float>
    {

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

        const int MAX_POSE_NUM = 100;
        public int KeypointsCount { get; private set; }

        // regressors / points
        protected float[,] output0;

        // classificators / scores
        protected float[] output1 = new float[896];

        private SsdAnchor[] anchors;
        // private List<Result> results = new List<Result>();
        private SortedSet<Result> results = new SortedSet<Result>();


        public PoseDetect(string modelPath) : base(modelPath, true)
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

            // Get Keypoint Mode
            var odim0 = interpreter.GetOutputTensorInfo(0).shape;
            KeypointsCount = (odim0[2] - 4) / 2;
        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, input0);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }

        public async UniTask<Result> InvokeAsync(Texture inputTex, CancellationToken cancellationToken, PlayerLoopTiming timing = PlayerLoopTiming.Update)
        {
            await ToTensorAsync(inputTex, input0, cancellationToken);
            await UniTask.SwitchToThreadPool();

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);

            var results = GetResults();

            await UniTask.SwitchToMainThread(timing, cancellationToken);
            return results;
        }

        public Result GetResults(float scoreThreshold = 0.5f, float iouThreshold = 0.3f)
        {
            results.Clear();

            int keypointsCount = KeypointsCount;

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

                var keypoints = new Vector2[keypointsCount];
                for (int j = 0; j < keypointsCount; j++)
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

            // No result
            if (results.Count == 0)
            {
                return Result.Negative;
            }

            return results.First();
            // return NonMaxSuppression(results, iouThreshold).First();
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

    public sealed class PoseDetectUpperBody : PoseDetect
    {
        public PoseDetectUpperBody(string modelPath) : base(modelPath)
        {
            // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
            // 4 - 11 are 4 keypoints x and y coordinates: x0,y0,x1,y1,x2,y2,x3,y3
            output0 = new float[896, 12];
        }
    }

    public sealed class PoseDetectFullBody : PoseDetect
    {
        public PoseDetectFullBody(string modelPath) : base(modelPath)
        {
            // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
            // 4 - 11 are 2 keypoints x and y coordinates: x0,y0,x1,y1
            output0 = new float[896, 8];
        }
    }
}
