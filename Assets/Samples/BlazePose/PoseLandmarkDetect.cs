using System.Diagnostics;
using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// https://google.github.io/mediapipe/solutions/pose.html
    /// 
    /// pose_landmark_upper_body_topology
    /// https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_upper_body_topology.svg
    /// </summary>
    public sealed class PoseLandmarkDetect : BaseImagePredictor<float>
    {
        public class Result
        {
            public float score;
            // x, y, z, w = visibility
            public Vector4[] viewportLandmarks;
            public Vector4[] worldLandmarks;
        }

        [System.Serializable]
        public class Options
        {
            public bool useWorldLandmarks = true;
            public bool useFilter = true;
            public Vector3 filterVelocityScale = new Vector3(10, 10, 2);

            private Vector3 cachedFilterVelocityScale;

            public bool CheckFilterUpdated()
            {
                bool isUpdated = cachedFilterVelocityScale != filterVelocityScale;
                cachedFilterVelocityScale = filterVelocityScale;
                return isUpdated;
            }
        }

        public const int LandmarkCount = 33;
        // A pair of indexes
        public static readonly int[] Connections = new int[]
        {
            // the same as Upper Body 
            0, 1, 1, 2, 2, 3, 3, 7, 0, 4, 4, 5, 5, 6, 6, 8, 9, 10, 11, 12, 11, 13, 13, 15, 15, 17, 15, 19, 15, 21, 17, 19, 12, 14, 14, 16, 16, 18, 16, 20, 16, 22, 18, 20, 11, 23, 12, 24, 23, 24,
            // left leg
            24, 26, 26, 28, 28, 32, 32, 30, 30, 28,
            // right leg
            23, 25, 25, 27, 27, 31, 31, 29, 29, 27,
        };

        // ld_3d
        private readonly float[] output0 = new float[195];
        // output_poseflag
        private readonly float[] output1 = new float[1];
        // output_segmentation, not in use
        // private float[,] output2 = new float[128, 128]; 
        // output_heatmap
        private readonly float[,,] output3 = new float[64, 64, 39];
        // world_3d
        private readonly float[] output4 = new float[117];

        private readonly Result result;
        private readonly Stopwatch stopwatch;
        private readonly RelativeVelocityFilter3D[] filters;
        private readonly Options options;
        private Matrix4x4 cropMatrix;

        // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
        public Vector2 PoseShift { get; set; } = new Vector2(0, 0);
        public Vector2 PoseScale { get; set; } = new Vector2(1.5f, 1.5f);
        public Matrix4x4 CropMatrix => cropMatrix;


        public PoseLandmarkDetect(string modelPath, Options options) : base(modelPath, true)
        {
            result = new Result()
            {
                score = 0,
                viewportLandmarks = new Vector4[LandmarkCount],
                worldLandmarks = new Vector4[LandmarkCount],
            };

            this.options = options ?? new Options();

            // Init filters
            filters = new RelativeVelocityFilter3D[LandmarkCount];
            const int windowSize = 5;
            const float velocityScale = 10;
            const RelativeVelocityFilter.DistanceEstimationMode mode = RelativeVelocityFilter.DistanceEstimationMode.LegacyTransition;
            for (int i = 0; i < LandmarkCount; i++)
            {
                filters[i] = new RelativeVelocityFilter3D(windowSize, velocityScale, mode);
            }
            UpdateFilterScale(options.filterVelocityScale);
            stopwatch = Stopwatch.StartNew();
        }

        public override void Invoke(Texture inputTex)
        {
            throw new System.NotImplementedException("Use Invoke(Texture, PalmDetect.Result)");
        }

        public void Invoke(Texture inputTex, PoseDetect.Result pose)
        {
            var resizeOptions = (inputTex is WebCamTexture)
                ? this.resizeOptions.GetModifedForWebcam((WebCamTexture)inputTex)
                : this.resizeOptions;

            cropMatrix = CalcCropMatrix(ref pose, ref resizeOptions);

            RenderTexture rt = resizer.Resize(
               inputTex, resizeOptions.width, resizeOptions.height, true,
               cropMatrix,
               TextureResizer.GetTextureST(inputTex, resizeOptions));
            ToTensor(rt, input0, false);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
            // interpreter.GetOutputTensorData(2, output2);// not in use
            if (options.useWorldLandmarks)
            {
                interpreter.GetOutputTensorData(3, output3);
                interpreter.GetOutputTensorData(4, output4);
            }
        }

        public async UniTask<Result> InvokeAsync(Texture inputTex, PoseDetect.Result pose, CancellationToken cancellationToken, PlayerLoopTiming timing)
        {
            var resizeOptions = (inputTex is WebCamTexture)
                ? base.resizeOptions.GetModifedForWebcam((WebCamTexture)inputTex)
                : base.resizeOptions;

            cropMatrix = CalcCropMatrix(ref pose, ref resizeOptions);
            RenderTexture rt = resizer.Resize(
              inputTex, resizeOptions.width, resizeOptions.height, true,
              cropMatrix,
              TextureResizer.GetTextureST(inputTex, resizeOptions));
            await ToTensorAsync(rt, input0, false, cancellationToken);
            await UniTask.SwitchToThreadPool();

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
            if (options.useWorldLandmarks)
            {
                interpreter.GetOutputTensorData(3, output3);
                interpreter.GetOutputTensorData(4, output4);
            }
            var result = GetResult();
            await UniTask.SwitchToMainThread(timing, cancellationToken);

            return result;
        }

        public Result GetResult()
        {
            // Normalize 0 ~ 255 => 0.0 ~ 1.0
            const float SCALE = 1f / 255f;
            var mtx = cropMatrix.inverse;

            // https://google.github.io/mediapipe/solutions/pose.html#output
            // The magnitude of z uses roughly the same scale as x.
            float xScale = Mathf.Abs(mtx.lossyScale.x);
            float zScale = SCALE * xScale * xScale;

            result.score = output1[0];

            Vector2 min = new Vector2(float.MaxValue, float.MaxValue);
            Vector2 max = new Vector2(float.MinValue, float.MinValue);

            int dimensions = output0.Length / LandmarkCount;

            for (int i = 0; i < LandmarkCount; i++)
            {
                Vector4 p = mtx.MultiplyPoint3x4(new Vector3(
                    output0[i * dimensions] * SCALE,
                    1f - output0[i * dimensions + 1] * SCALE,
                    output0[i * dimensions + 2] * zScale
                ));
                p.w = output0[i * dimensions + 3];
                result.viewportLandmarks[i] = p;

                if (p.x < min.x) { min.x = p.x; }
                if (p.x > max.x) { max.x = p.x; }
                if (p.y < min.y) { min.y = p.y; }
                if (p.y > max.y) { max.y = p.y; }
            }

            if (options.useFilter)
            {
                if (options.CheckFilterUpdated())
                {
                    UpdateFilterScale(options.filterVelocityScale);
                }

                // Apply filters
                double timestamp = stopwatch.Elapsed.TotalSeconds;
                Vector2 size = max - min;
                float valueScale = 1f / ((size.x + size.y) / 2);
                for (int i = 0; i < LandmarkCount; i++)
                {
                    Vector4 joint = result.viewportLandmarks[i];
                    Vector4 filterd = filters[i].Apply(timestamp, valueScale, (Vector3)joint);
                    filterd.w = joint.w;
                    result.viewportLandmarks[i] = filterd;
                }
            }

            if (options.useWorldLandmarks)
            {
                CalcWorldLandmarks(result);
            }

            return result;
        }

        private void CalcWorldLandmarks(Result result)
        {
            int dimensions = output4.Length / LandmarkCount;
            for (int i = 0; i < LandmarkCount; i++)
            {
                result.worldLandmarks[i] = new Vector4(
                    output4[i * dimensions],
                    -output4[i * dimensions + 1],
                    output4[i * dimensions + 2],
                    result.viewportLandmarks[i].w
                );
            }
        }

        private static Rect AlignmentPointsToRect(in Vector2 center, in Vector2 scale)
        {
            float boxSize = Mathf.Sqrt(
                (scale.x - center.x) * (scale.x - center.x)
                + (scale.y - center.y) * (scale.y - center.y)
            ) * 2f;
            return new Rect(
                center.x - boxSize / 2,
                center.y - boxSize / 2,
                boxSize,
                boxSize);
        }

        private static float CalcRotationDegree(in Vector2 a, in Vector2 b)
        {
            const float RAD_90 = 90f * Mathf.PI / 180f;
            var vec = a - b;
            return -(RAD_90 + Mathf.Atan2(vec.y, vec.x)) * Mathf.Rad2Deg;
        }

        private void UpdateFilterScale(Vector3 scale)
        {
            foreach (var f in filters)
            {
                f.VelocityScale = scale;
            }
        }

        private Matrix4x4 CalcCropMatrix(ref PoseDetect.Result pose, ref TextureResizer.ResizeOptions options)
        {
            float rotation = CalcRotationDegree(pose.keypoints[0], pose.keypoints[1]);
            var rect = AlignmentPointsToRect(pose.keypoints[0], pose.keypoints[1]);
            return RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = rect,
                rotationDegree = rotation,
                shift = PoseShift,
                scale = PoseScale,
                cameraRotationDegree = -options.rotationDegree,
                mirrorHorizontal = options.mirrorHorizontal,
                mirrorVertiacal = options.mirrorVertical,
            });
        }

        public static PoseDetect.Result LandmarkToDetection(Result result)
        {
            Vector2 hip = (result.viewportLandmarks[24] + result.viewportLandmarks[23]) / 2f;
            Vector2 nose = result.viewportLandmarks[0];
            Vector2 aboveHead = hip + (nose - hip) * 1.2f;
            // Y Flipping
            hip.y = 1f - hip.y;
            aboveHead.y = 1f - aboveHead.y;

            return new PoseDetect.Result()
            {
                score = result.score,
                keypoints = new Vector2[] { hip, aboveHead },
            };
        }
    }
}
