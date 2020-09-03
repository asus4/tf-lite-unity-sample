using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TensorFlowLite
{
    public class FaceMesh : BaseImagePredictor<float>
    {
        public class Result
        {
            public float score;
            public Vector3[] keypoints;
        }
        public const int KEYPOINT_COUNT = 468;
        private float[,] output0 = new float[KEYPOINT_COUNT, 3]; // keypoint
        private float[] output1 = new float[1]; // flag

        private Result result;
        private Matrix4x4 cropMatrix;

        public Vector2 FaceShift { get; set; } = new Vector2(0, 0.1f);
        public Vector2 FaceScale { get; set; } = new Vector2(1.5f, 1.5f);
        public Matrix4x4 CropMatrix => cropMatrix;


        public FaceMesh(string modelPath) : base(modelPath, true)
        {
            result = new Result()
            {
                score = 0,
                keypoints = new Vector3[KEYPOINT_COUNT],
            };
        }

        public override void Invoke(Texture inputTex)
        {
            throw new System.NotImplementedException("Use Invoke(Texture inputTex, FaceDetect.Result palm)");
        }

        public void Invoke(Texture inputTex, FaceDetect.Result face)
        {
            var options = (inputTex is WebCamTexture)
                ? resizeOptions.GetModifedForWebcam((WebCamTexture)inputTex)
                : resizeOptions;

            cropMatrix = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = face.rect,
                rotationDegree = CalcFaceRotation(ref face) * Mathf.Rad2Deg,
                shift = FaceShift,
                scale = FaceScale,
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

        public Result GetResult()
        {
            const float SCALE = 1f / 192f;
            var mtx = cropMatrix.inverse;

            result.score = output1[0];
            for (int i = 0; i < KEYPOINT_COUNT; i++)
            {
                result.keypoints[i] = mtx.MultiplyPoint3x4(new Vector3(
                    output0[i, 0] * SCALE,
                    1f - output0[i, 1] * SCALE,
                    output0[i, 2] * SCALE
                ));
            }
            return result;
        }

        private static float CalcFaceRotation(ref FaceDetect.Result detection)
        {
            var vec = detection.rightEye - detection.leftEye;
            return -Mathf.Atan2(vec.y, vec.x);
        }
    }
}
