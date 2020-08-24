using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace TensorFlowLite
{
    public class HandLandmarkDetect : BaseImagePredictor<float>
    {
        public class Result
        {
            public float score;
            public Vector3[] joints;
        }

        public enum Dimension
        {
            TWO,
            THREE,
        }

        public const int JOINT_COUNT = 21;

        private float[] output0 = new float[JOINT_COUNT * 2]; // keypoint
        private float[] output1 = new float[1]; // hand flag
        private Result result;
        private Matrix4x4 cropMatrix;

        public Dimension Dim { get; private set; }
        public Vector2 PalmShift { get; set; } = new Vector2(0, 0.2f);
        public Vector2 PalmScale { get; set; } = new Vector2(2.8f, 2.8f);
        public Matrix4x4 CropMatrix => cropMatrix;

        public HandLandmarkDetect(string modelPath) : base(modelPath, true)
        {
            var out0info = interpreter.GetOutputTensorInfo(0);
            switch (out0info.shape[1])
            {
                case JOINT_COUNT * 2:
                    Dim = Dimension.TWO;
                    break;
                case JOINT_COUNT * 3:
                    Dim = Dimension.THREE;
                    break;
                default:
                    throw new System.NotSupportedException();
            }
            output0 = new float[out0info.shape[1]];

            result = new Result()
            {
                score = 0,
                joints = new Vector3[JOINT_COUNT],
            };
        }

        public override void Invoke(Texture inputTex)
        {
            throw new System.NotImplementedException("Use Invoke(Texture inputTex, PalmDetect.Palm palm)");
        }

        public void Invoke(Texture inputTex, PalmDetect.Result palm)
        {
            var options = resizeOptions;
            if (inputTex is WebCamTexture)
            {
                options = TextureResizer.ModifyOptionForWebcam(options, (WebCamTexture)inputTex);
            }

            float rotation = CalcHandRotation(ref palm) * Mathf.Rad2Deg;
            var mat = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = palm.rect,
                rotationDegree = rotation,
                shift = PalmShift,
                scale = PalmScale,
                cameraRotationDegree = 0,
                mirrorHorizontal = !options.mirrorHorizontal,
                mirrorVertiacal = options.mirrorVertical,
            });
            cropMatrix = resizer.VertexTransfrom = mat;

            resizer.UVRect = TextureResizer.GetTextureST(inputTex, options);
            RenderTexture rt = resizer.ApplyResize(inputTex, options.width, options.height, true);
            ToTensor(rt, input0, false);

            //
            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }

        public Result GetResult()
        {
            // Normalize 0 ~ 255 => 0.0 ~ 1.0
            const float SCALE = 1f / 255f;
            var mtx = cropMatrix.inverse;

            result.score = output1[0];
            if (Dim == Dimension.TWO)
            {
                for (int i = 0; i < JOINT_COUNT; i++)
                {
                    result.joints[i] = mtx.MultiplyPoint3x4(new Vector3(
                        output0[i * 2] * SCALE,
                        1f - output0[i * 2 + 1] * SCALE,
                        0
                    ));
                }
            }
            else
            {
                for (int i = 0; i < JOINT_COUNT; i++)
                {
                    result.joints[i] = mtx.MultiplyPoint3x4(new Vector3(
                        output0[i * 3] * SCALE,
                        1f - output0[i * 3 + 1] * SCALE,
                        output0[i * 3 + 2] * SCALE
                    ));
                }
            }
            return result;
        }

        private static float CalcHandRotation(ref PalmDetect.Result detection)
        {
            // Rotation based on Center of wrist - Middle finger
            const float RAD_90 = 90f * Mathf.PI / 180f;
            var vec = detection.keypoints[0] - detection.keypoints[2];
            return -(RAD_90 + Mathf.Atan2(vec.y, vec.x));
        }
    }
}
