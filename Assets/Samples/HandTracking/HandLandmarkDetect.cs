using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using Cysharp.Threading.Tasks;


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

        public static readonly int[] CONNECTIONS = new int[] { 0, 1, 1, 2, 2, 3, 3, 4, 0, 5, 5, 6, 6, 7, 7, 8, 5, 9, 9, 10, 10, 11, 11, 12, 9, 13, 13, 14, 14, 15, 15, 16, 13, 17, 0, 17, 17, 18, 18, 19, 19, 20, };
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
            var options = (inputTex is WebCamTexture)
                ? resizeOptions.GetModifedForWebcam((WebCamTexture)inputTex)
                : resizeOptions;

            cropMatrix = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = palm.rect,
                rotationDegree = CalcHandRotation(ref palm) * Mathf.Rad2Deg,
                shift = PalmShift,
                scale = PalmScale,
                cameraRotationDegree = -options.rotationDegree,
                mirrorHorizontal = options.mirrorHorizontal,
                mirrorVertiacal = options.mirrorVertical,
            });

            RenderTexture rt = resizer.Resize(
                inputTex, options.width, options.height, true,
                cropMatrix,
                TextureResizer.GetTextureST(inputTex, options));
            ToTensor(rt, input0, false);

            //
            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }

        public async UniTask<Result> InvokeAsync(Texture inputTex, PalmDetect.Result palm, CancellationToken cancellationToken)
        {
            var options = (inputTex is WebCamTexture)
               ? resizeOptions.GetModifedForWebcam((WebCamTexture)inputTex)
               : resizeOptions;
            cropMatrix = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = palm.rect,
                rotationDegree = CalcHandRotation(ref palm) * Mathf.Rad2Deg,
                shift = PalmShift,
                scale = PalmScale,
                cameraRotationDegree = -options.rotationDegree,
                mirrorHorizontal = options.mirrorHorizontal,
                mirrorVertiacal = options.mirrorVertical,
            });

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

            var result = GetResult();
            await UniTask.SwitchToMainThread(cancellationToken);
            return result;
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
            const float RAD_90 = 90f * Mathf.Deg2Rad;
            var vec = detection.keypoints[0] - detection.keypoints[2];
            return -(RAD_90 + Mathf.Atan2(vec.y, vec.x));
        }
    }
}
