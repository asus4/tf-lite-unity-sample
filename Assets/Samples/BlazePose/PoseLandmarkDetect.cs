using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;

using UnityEngine;
using Unity.Mathematics;

namespace TensorFlowLite
{
    /// <summary>
    /// pose_landmark_upper_body_topology
    /// https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_upper_body_topology.svg
    /// </summary>
    public class PoseLandmarkDetect : BaseImagePredictor<float>
    {
        public class Result
        {
            public float score;
            public Vector3[] joints;
        }

        public const int JOINT_COUNT = 25;
        // A pair of indexes
        public static readonly int[] CONNECTIONS = new int[] { 0, 1, 1, 2, 2, 3, 3, 7, 0, 4, 4, 5, 5, 6, 6, 8, 9, 10, 11, 12, 11, 13, 13, 15, 15, 17, 15, 19, 15, 21, 17, 19, 12, 14, 14, 16, 16, 18, 16, 20, 16, 22, 18, 20, 11, 23, 12, 24, 23, 24, };

        private float[] output0 = new float[124]; // ld_3d
        private float[] output1 = new float[1]; // output_poseflag
        // private float[,] output2 = new float[128, 128]; // output_segmentation, not in use
        private Result result;
        private Matrix4x4 cropMatrix;
        private Stopwatch stopwatch;
        private RelativeVelocityFilter3D[] filter;

        // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
        public Vector2 PoseShift { get; set; } = new Vector2(0, 0);
        public Vector2 PoseScale { get; set; } = new Vector2(1.5f, 1.5f);
        public Matrix4x4 CropMatrix => cropMatrix;

        public PoseLandmarkDetect(string modelPath) : base(modelPath, true)
        {
            result = new Result()
            {
                score = 0,
                joints = new Vector3[JOINT_COUNT],
            };

            // Init filters
            filter = new RelativeVelocityFilter3D[JOINT_COUNT];
            const int windowSize = 5;
            const float velocityScale = 10;
            const RelativeVelocityFilter.DistanceEstimationMode mode = RelativeVelocityFilter.DistanceEstimationMode.LegacyTransition;
            for (int i = 0; i < JOINT_COUNT; i++)
            {
                filter[i] = new RelativeVelocityFilter3D(windowSize, velocityScale, mode);
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

            // float rotation = CalcRotationDegree(ref pose);
            var rect = AlignmentPointsRect(ref pose);
            cropMatrix = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = rect,
                rotationDegree = 180,
                shift = PoseShift,
                scale = PoseScale,
                cameraRotationDegree = -options.rotationDegree,
                mirrorHorizontal = options.mirrorHorizontal,
                mirrorVertiacal = options.mirrorVertical,
            });

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

        public Result GetResult(bool useFilter = true)
        {
            // Normalize 0 ~ 255 => 0.0 ~ 1.0
            const float SCALE = 1f / 255f;
            var mtx = cropMatrix.inverse;

            result.score = output1[0];

            Vector2 min = new Vector2(float.MaxValue, float.MaxValue);
            Vector2 max = new Vector2(float.MinValue, float.MinValue);

            for (int i = 0; i < JOINT_COUNT; i++)
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
                for (int i = 0; i < JOINT_COUNT; i++)
                {
                    Vector3 p = result.joints[i];
                    result.joints[i] = filter[i].Apply(timestamp, valueScale, p);
                }
            }

            return result;
        }


        private static float CalcRotationDegree(ref PoseDetect.Result pose)
        {
            // Calc rotation based on 
            // Center of Hip and Center of shoulder
            const float RAD_90 = 90f * Mathf.PI / 180f;
            var vec = pose.keypoints[0] - pose.keypoints[2];
            return -(RAD_90 + Mathf.Atan2(vec.y, vec.x)) * Mathf.Rad2Deg;
        }

        // AlignmentPointsRectsCalculator from MediaPipe
        private static Rect AlignmentPointsRect(ref PoseDetect.Result pose)
        {
            float2 center = pose.keypoints[2];
            float2 scale = pose.keypoints[3];
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


    }
}
