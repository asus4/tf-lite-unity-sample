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
        public static readonly int[] CONNECTIONS = new int[] { 0, 1, 1, 2, 2, 3, 3, 7, 0, 4, 4, 5, 5, 6, 6, 8, 9, 10, 11, 12, 11, 13, 13, 15, 15, 17, 15, 19, 15, 21, 17, 19, 12, 14, 14, 16, 16, 18, 16, 20, 16, 22, 18, 20, 11, 23, 12, 24, 23, 24, };

        private float[] output0 = new float[124]; // ld_3d
        private float[] output1 = new float[1]; // output_poseflag
        // private float[,] output2 = new float[128, 128]; // output_segmentation, not in use
        private Result result;
        private Matrix4x4 cropMatrix;
        private Stopwatch stopwatch;
        private RelativeVelocityFilter[] filterX;
        private RelativeVelocityFilter[] filterY;
        private RelativeVelocityFilter[] filterZ;

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
            filterX = new RelativeVelocityFilter[JOINT_COUNT];
            filterY = new RelativeVelocityFilter[JOINT_COUNT];
            filterZ = new RelativeVelocityFilter[JOINT_COUNT];
            for (int i = 0; i < JOINT_COUNT; i++)
            {
                filterX[i] = new RelativeVelocityFilter(5, 10, RelativeVelocityFilter.DistanceEstimationMode.kForceCurrentScale);
                filterY[i] = new RelativeVelocityFilter(5, 10, RelativeVelocityFilter.DistanceEstimationMode.kForceCurrentScale);
                filterZ[i] = new RelativeVelocityFilter(5, 10, RelativeVelocityFilter.DistanceEstimationMode.kForceCurrentScale);
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
                ? TextureResizer.ModifyOptionForWebcam(resizeOptions, (WebCamTexture)inputTex)
                : resizeOptions;

            // float rotation = CalcRotationDegree(ref pose);
            const float rotation = 180;
            var rect = AlignmentPointsRect(ref pose);
            var mat = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = rect,
                rotationDegree = rotation,
                shift = PoseShift,
                scale = PoseScale,
                cameraRotationDegree = 0,
                mirrorHorizontal = !options.mirrorHorizontal,
                mirrorVertiacal = options.mirrorVertical,
            });
            cropMatrix = resizer.VertexTransfrom = mat;
            resizer.UVRect = TextureResizer.GetTextureST(inputTex, options);
            RenderTexture rt = resizer.ApplyResize(inputTex, options.width, options.height, true);
            ToTensor(rt, input0, false);

            // cropMatrix = Matrix4x4.identity;
            // ToTensor(inputTex, input0);

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
                UnityEngine.Debug.Log("filtering");
                // Apply filters
                const long TICKS_TO_NANO = 10;
                long timestampNS = stopwatch.Elapsed.Ticks * TICKS_TO_NANO;
                Vector2 size = max - min;
                float valueScale = 1f / ((size.x + size.y) / 2);
                for (int i = 0; i < JOINT_COUNT; i++)
                {
                    Vector3 p = result.joints[i];
                    p.x = filterX[i].Apply(timestampNS, valueScale, p.x);
                    p.y = filterY[i].Apply(timestampNS, valueScale, p.y);
                    p.z = filterZ[i].Apply(timestampNS, valueScale, p.z);
                    result.joints[i] = p;
                }
            }
            else
            {
                UnityEngine.Debug.Log("NO filter");
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
