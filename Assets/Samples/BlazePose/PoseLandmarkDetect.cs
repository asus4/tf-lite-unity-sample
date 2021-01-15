using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using UnityEngine;
using Cysharp.Threading.Tasks;

namespace TensorFlowLite
{
    /// <summary>
    /// pose_landmark_upper_body_topology
    /// https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_upper_body_topology.svg
    /// </summary>
    public abstract class PoseLandmarkDetect : BaseImagePredictor<float>
    {
        public class Result
        {
            public float score;
            public Vector3[] joints;
        }

        public abstract int JointCount { get; }
        // A pair of indexes
        public abstract int[] Connections { get; }

        protected float[] output0; // ld_3d
        private float[] output1 = new float[1]; // output_poseflag
        // private float[,] output2 = new float[128, 128]; // output_segmentation, not in use
        private Result result;
        private Matrix4x4 cropMatrix;
        private Stopwatch stopwatch;
        private RelativeVelocityFilter2D[] filters;

        // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
        public Vector2 PoseShift { get; set; } = new Vector2(0, 0);
        public Vector2 PoseScale { get; set; } = new Vector2(1.5f, 1.5f);
        public float FilterVelocityScale
        {
            get
            {
                return filters[0].VelocityScale;
            }
            set
            {
                foreach (var f in filters)
                {
                    f.VelocityScale = value;
                }
            }
        }

        public Matrix4x4 CropMatrix => cropMatrix;

        public PoseLandmarkDetect(string modelPath) : base(modelPath, true)
        {
            result = new Result()
            {
                score = 0,
                joints = new Vector3[JointCount],
            };

            // Init filters
            filters = new RelativeVelocityFilter2D[JointCount];
            const int windowSize = 5;
            const float velocityScale = 10;
            const RelativeVelocityFilter.DistanceEstimationMode mode = RelativeVelocityFilter.DistanceEstimationMode.LegacyTransition;
            for (int i = 0; i < JointCount; i++)
            {
                filters[i] = new RelativeVelocityFilter2D(windowSize, velocityScale, mode);
            }
            stopwatch = Stopwatch.StartNew();
        }

        public override void Invoke(Texture inputTex)
        {
            throw new System.NotImplementedException("Use Invoke(Texture, PalmDetect.Result)");
        }

        public void Invoke(Texture inputTex, PoseDetect.Result pose)
        {
            var options = (inputTex is WebCamTexture)
                ? resizeOptions.GetModifedForWebcam((WebCamTexture)inputTex)
                : resizeOptions;

            cropMatrix = CalcCropMatrix(ref pose, ref options);

            RenderTexture rt = resizer.Resize(
               inputTex, options.width, options.height, true,
               cropMatrix,
               TextureResizer.GetTextureST(inputTex, options));
            ToTensor(rt, input0, false);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);

            // interpreter.GetOutputTensorData(2, output2);// not in use
        }

        public async UniTask<Result> InvokeAsync(Texture inputTex, PoseDetect.Result pose, bool useFilter, CancellationToken cancellationToken, PlayerLoopTiming timing)
        {
            var options = (inputTex is WebCamTexture)
                ? resizeOptions.GetModifedForWebcam((WebCamTexture)inputTex)
                : resizeOptions;

            cropMatrix = CalcCropMatrix(ref pose, ref options);
            RenderTexture rt = resizer.Resize(
              inputTex, options.width, options.height, true,
              cropMatrix,
              TextureResizer.GetTextureST(inputTex, options));
            await ToTensorAsync(rt, input0, false, cancellationToken);
            await UniTask.SwitchToThreadPool();

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);

            var result = GetResult(useFilter);
            await UniTask.SwitchToMainThread(timing, cancellationToken);

            return result;
        }

        public Result GetResult(bool useFilter = true)
        {
            // Normalize 0 ~ 255 => 0.0 ~ 1.0
            const float SCALE = 1f / 255f;
            var mtx = cropMatrix.inverse;

            result.score = output1[0];

            Vector2 min = new Vector2(float.MaxValue, float.MaxValue);
            Vector2 max = new Vector2(float.MinValue, float.MinValue);

            for (int i = 0; i < JointCount; i++)
            {
                Vector3 p = mtx.MultiplyPoint3x4(new Vector3(
                    output0[i * 4] * SCALE,
                    1f - output0[i * 4 + 1] * SCALE,
                    output0[i * 4 + 2] * SCALE
                ));
                result.joints[i] = p;

                if (p.x < min.x) { min.x = p.x; }
                if (p.x > max.x) { max.x = p.x; }
                if (p.y < min.y) { min.y = p.y; }
                if (p.y > max.y) { max.y = p.y; }
            }

            if (useFilter)
            {
                // Apply filters
                double timestamp = stopwatch.Elapsed.TotalSeconds;
                Vector2 size = max - min;
                float valueScale = 1f / ((size.x + size.y) / 2);
                for (int i = 0; i < JointCount; i++)
                {
                    Vector3 p = result.joints[i];
                    Vector2 filtered = filters[i].Apply(timestamp, valueScale, p);
                    result.joints[i] = new Vector3(filtered.x, filtered.y, p.z);
                }
            }

            return result;
        }

        protected static Rect AlignmentPointsToRect(in Vector2 center, in Vector2 scale)
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

        protected static float CalcRotationDegree(in Vector2 a, in Vector2 b)
        {
            const float RAD_90 = 90f * Mathf.PI / 180f;
            var vec = a - b;
            return -(RAD_90 + Mathf.Atan2(vec.y, vec.x)) * Mathf.Rad2Deg;
        }

        protected abstract Matrix4x4 CalcCropMatrix(ref PoseDetect.Result pose, ref TextureResizer.ResizeOptions options);
    }

    public sealed class PoseLandmarkDetectUpperBody : PoseLandmarkDetect
    {
        public override int JointCount => 25;
        public override int[] Connections => CONNECTIONS;

        // https://google.github.io/mediapipe/solutions/pose
        private static readonly int[] CONNECTIONS = new int[] { 0, 1, 1, 2, 2, 3, 3, 7, 0, 4, 4, 5, 5, 6, 6, 8, 9, 10, 11, 12, 11, 13, 13, 15, 15, 17, 15, 19, 15, 21, 17, 19, 12, 14, 14, 16, 16, 18, 16, 20, 16, 22, 18, 20, 11, 23, 12, 24, 23, 24, };

        public PoseLandmarkDetectUpperBody(string modelPath) : base(modelPath)
        {
            output0 = new float[124]; // ld_3d
            PoseShift = new Vector2(0, 0);
            PoseScale = new Vector2(1.5f, 1.5f);
        }

        protected override Matrix4x4 CalcCropMatrix(ref PoseDetect.Result pose, ref TextureResizer.ResizeOptions options)
        {
            float rotation = CalcRotationDegree(pose.keypoints[2], pose.keypoints[3]);
            var rect = AlignmentPointsToRect(pose.keypoints[2], pose.keypoints[3]);
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
    }

    public sealed class PoseLandmarkDetectFullBody : PoseLandmarkDetect
    {
        public override int JointCount => 33;
        public override int[] Connections => CONNECTIONS;

        // https://developers.google.com/ml-kit/vision/pose-detection
        private static readonly int[] CONNECTIONS = new int[] {
            // the same as Upper Body 
            0, 1, 1, 2, 2, 3, 3, 7, 0, 4, 4, 5, 5, 6, 6, 8, 9, 10, 11, 12, 11, 13, 13, 15, 15, 17, 15, 19, 15, 21, 17, 19, 12, 14, 14, 16, 16, 18, 16, 20, 16, 22, 18, 20, 11, 23, 12, 24, 23, 24,
            // left leg
            24, 26, 26, 28, 28, 32, 32, 30, 30, 28,
            // right leg
            23, 25, 25, 27, 27, 31, 31, 29, 29, 27,
         };
        public PoseLandmarkDetectFullBody(string modelPath) : base(modelPath)
        {
            output0 = new float[156]; // ld_3d
            PoseShift = new Vector2(0, 0f);
            PoseScale = new Vector2(1.8f, 1.8f);
        }

        protected override Matrix4x4 CalcCropMatrix(ref PoseDetect.Result pose, ref TextureResizer.ResizeOptions options)
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
    }
}
