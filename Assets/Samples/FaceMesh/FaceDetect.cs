using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace TensorFlowLite
{
    public class FaceDetect : BaseVisionTask
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
            /// <summary>
            /// Model space bounding box top-left:[0,0], Normalized to [0, 1] 
            /// </summary>
            public Rect rect;
            public Vector2[] keypoints;

            public Vector2 RightEye => keypoints[(int)KeyPoint.RightEye];
            public Vector2 LeftEye => keypoints[(int)KeyPoint.LeftEye];
        }

        private const int KEY_POINT_COUNT = 6;
        private const int MAX_FACE_NUM = 100;

        // regressors / points
        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 15 are 6 keypoint x and y coordinates: x0,y0,x1,y1,x2,y2,x3,y3
        private readonly float[,] output0 = new float[896, 16];

        // classificators / scores
        private readonly float[] output1 = new float[896];

        private readonly SsdAnchor[] anchors;
        private readonly List<Result> results = new();
        private readonly List<Result> filteredResults = new();

        public Matrix4x4 InputTransformMatrix { get; private set; } = Matrix4x4.identity;

        public FaceDetect(string modelPath)
        {
            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AutoAddDelegate(TfLiteDelegateType.GPU, typeof(float));
            Load(FileUtil.LoadFile(modelPath), interpreterOptions);
            AspectMode = AspectMode.Fill;

            anchors = SsdAnchorsCalculator.Generate(new()
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
            });
            Assert.AreEqual(896, anchors.Length, $"Anchors count must be 896, but was {anchors.Length}");
        }

        protected override void PreProcess(Texture texture)
        {
            InputTransformMatrix = textureToTensor.GetAspectScaledMatrix(texture, AspectMode);
            var input = textureToTensor.Transform(texture, InputTransformMatrix);
            interpreter.SetInputTensorData(inputTensorIndex, input);
        }

        protected override void PostProcess()
        {
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

                var keypoints = new Vector2[KEY_POINT_COUNT];
                for (int j = 0; j < KEY_POINT_COUNT; j++)
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
            filteredResults.Clear();
            // FIXME: Suppress LinQ sort allocation
            // Use sorted list
            foreach (Result original in results.OrderByDescending(o => o.score))
            {
                bool ignoreCandidate = false;
                foreach (Result newResult in filteredResults)
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
                    filteredResults.Add(original);
                    if (filteredResults.Count >= MAX_FACE_NUM)
                    {
                        break;
                    }
                }
            }
            return filteredResults;
        }
    }
}
