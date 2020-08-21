using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// pose_landmark_upper_body_topology
    /// https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_upper_body_topology.svg
    /// </summary>
    public class PoseLandmark : BaseImagePredictor<float>
    {
        public class Result
        {
            public float score;
            public Vector3[] joints;
        }

        public const int JOINT_COUNT = 25;

        private float[] output0 = new float[124]; // ld_3d
        private float[] output1 = new float[1]; // output_poseflag
        // private float[,] output2 = new float[128, 128]; // output_segmentation, not in use
        private Result result;
        private Matrix4x4 cropMatrix;

        public Vector2 PoseShift { get; set; } = new Vector2(0, 0);
        public Vector2 PoseScale { get; set; } = new Vector2(1.5f, 1.5f);

        public PoseLandmark(string modelPath) : base(modelPath, true)
        {
            result = new Result()
            {
                score = 0,
                joints = new Vector3[JOINT_COUNT],
            };
        }

        public override void Invoke(Texture inputTex)
        {
            throw new System.NotImplementedException("Use Invoke(Texture, PalmDetect.Result)");
        }

        public void Invoke(Texture inputTex, PoseDetect.Result pose)
        {
            var options = resizeOptions;

            cropMatrix = resizer.VertexTransfrom = CalcPoseMatrix(ref pose, PoseShift, PoseScale);

            resizer.UVRect = TextureResizer.GetTextureST(inputTex, options);
            RenderTexture rt = resizer.ApplyResize(inputTex, options.width, options.height, true);
            ToTensor(rt, input0, false);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
            // interpreter.GetOutputTensorData(2, output2);
        }

        public Result GetResult()
        {
            // Normalize 0 ~ 255 => 0.0 ~ 1.0
            const float SCALE = 1f / 255f;
            var mtx = cropMatrix.inverse;

            result.score = output1[0];

            for (int i = 0; i < JOINT_COUNT; i++)
            {
                result.joints[i] = mtx.MultiplyPoint3x4(new Vector3(
                    output0[i * 4],
                    output0[i * 4 + 1],
                    output0[i * 4 + 2]
                ) * SCALE);
            }

            return result;
        }

        private static readonly Matrix4x4 PUSH_MATRIX = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        private static readonly Matrix4x4 POP_MATRIX = Matrix4x4.Translate(new Vector3(-0.5f, -0.5f, 0));

        // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
        private static Matrix4x4 CalcPoseMatrix(ref PoseDetect.Result detection, Vector2 shift, Vector2 scale)
        {
            // Calc rotation based on 
            // Center of Hip and Center of shoulder
            const float RAD_90 = 90f * Mathf.PI / 180f;
            var vec = detection.keypoints[0] - detection.keypoints[2];
            Quaternion rotation = Quaternion.Euler(0, 0, (RAD_90 + Mathf.Atan2(vec.y, vec.x)) * Mathf.Rad2Deg);

            // Calc hand scale
            Vector2 size = Vector2.Scale(detection.rect.size, scale);

            // Calc hand center position
            Vector2 center = detection.rect.center + new Vector2(-0.5f, -0.5f);
            center = (Vector2)(rotation * center);
            center += (shift * size);
            center /= size;

            Matrix4x4 trs = Matrix4x4.TRS(
                               new Vector3(-center.x, -center.y, 0),
                               rotation,
                               new Vector3(1, 1, 1)
                            );
            return PUSH_MATRIX * trs * POP_MATRIX;
        }
    }
}
