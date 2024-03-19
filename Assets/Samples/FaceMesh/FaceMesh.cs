using System;
using System.Buffers;
using UnityEngine;
using DataType = TensorFlowLite.Interpreter.DataType;


namespace TensorFlowLite
{
    public class FaceMesh : BaseImagePredictor<float>
    {
        public class Result
        {
            public float score;
            /// <summary>
            /// 468 points in 3D space normalized 0-1
            /// </summary>
            public Vector3[] keypoints;
        }
        public const int KEYPOINT_COUNT = 468;
        private float[,] output0 = new float[KEYPOINT_COUNT, 3]; // keypoint
        private float[] output1 = new float[1]; // flag

        private Result result;
        private Matrix4x4 cropMatrix;

        private TensorToTexture debugInputTensorToTexture;

        public Vector2 FaceShift { get; set; } = new Vector2(0f, 0f);
        public Vector2 FaceScale { get; set; } = new Vector2(1.6f, 1.6f);
        public Matrix4x4 CropMatrix => cropMatrix;


        public RenderTexture InputTexture => debugInputTensorToTexture.OutputTexture;

        public FaceMesh(string modelPath) : base(modelPath, TfLiteDelegateType.GPU)
        {
            result = new Result()
            {
                score = 0,
                keypoints = new Vector3[KEYPOINT_COUNT],
            };

            debugInputTensorToTexture = new TensorToTexture(new()
            {
                compute = null,
                kernel = 0,
                width = width,
                height = height,
                channels = channels,
                inputType = DataType.Float32,
            });
        }

        public override void Invoke(Texture inputTex)
        {
            throw new NotImplementedException("Use Invoke(Texture inputTex, FaceDetect.Result palm)");
        }

        public void Invoke(Texture inputTex, FaceDetect.Result face)
        {
            cropMatrix = RectTransformationCalculator.CalcMatrix(new()
            {
                rect = face.rect,
                rotationDegree = CalcFaceRotation(ref face) * Mathf.Rad2Deg,
                shift = FaceShift,
                scale = FaceScale,
                mirrorHorizontal = resizeOptions.mirrorHorizontal,
                mirrorVertical = resizeOptions.mirrorVertical,
            });

            RenderTexture rt = resizer.Resize(
                inputTex, resizeOptions.width, resizeOptions.height, true,
                cropMatrix,
                TextureResizer.GetTextureST(inputTex, resizeOptions));
            ToTensor(rt, inputTensor, false);

            //
            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);

            debugInputTensorToTexture.Convert(inputTensor);
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

        /// <summary>
        /// Estimate face detection from face mesh
        /// </summary>
        /// <param name="faceMeshResult">a result from face mesh</param>
        /// <returns></returns>
        public static FaceDetect.Result LandmarkToDetection(Result faceMeshResult)
        {
            // Original index looks like a bug
            // rotation_vector_start_keypoint_index: 33  # Left side of left eye.
            // rotation_vector_end_keypoint_index: 133  # Right side of right eye.
            const int ID_RIGHT_EYE = 263;
            const int ID_LEFT_EYE = 33;

            Vector3[] buffer = ArrayPool<Vector3>.Shared.Rent(KEYPOINT_COUNT);
            Span<Vector3> keypoints = buffer.AsSpan(0, KEYPOINT_COUNT);
            for (int i = 0; i < KEYPOINT_COUNT; i++)
            {
                Vector3 v = faceMeshResult.keypoints[i];
                v.y = 1f - v.y;
                keypoints[i] = v;
            }

            Rect rect = RectExtension.GetBoundingBox(keypoints);
            Vector2 center = rect.center;
            float size = Mathf.Min(rect.width, rect.height);
            rect.Set(center.x - size * 0.5f, center.y - size * 0.5f, size, size);

            FaceDetect.Result detection = new()
            {
                score = faceMeshResult.score,
                rect = rect,
                keypoints = new Vector2[]
                {
                    keypoints[ID_RIGHT_EYE],
                    keypoints[ID_LEFT_EYE]
                },
            };
            ArrayPool<Vector3>.Shared.Return(buffer);
            return detection;
        }

        private static float CalcFaceRotation(ref FaceDetect.Result detection)
        {
            var vec = detection.RightEye - detection.LeftEye;
            return -Mathf.Atan2(vec.y, vec.x);
        }
    }
}
