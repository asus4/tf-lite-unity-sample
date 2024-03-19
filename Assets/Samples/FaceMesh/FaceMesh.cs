using System;
using System.Buffers;
using UnityEngine;
using DataType = TensorFlowLite.Interpreter.DataType;


namespace TensorFlowLite
{
    public class FaceMesh : BaseVisionTask
    {
        public class Result
        {
            public float score;
            /// <summary>
            /// 468 points in 3D space normalized 0-1
            /// </summary>
            public Vector3[] keypoints;

            /// <summary>
            /// Estimate face detection from face mesh
            /// </summary>
            /// <param name="faceMeshResult">a result from face mesh</param>
            /// <returns></returns>
            public FaceDetect.Result ToDetection()
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
                    Vector3 v = this.keypoints[i];
                    v.y = 1f - v.y;
                    keypoints[i] = v;
                }

                Rect rect = RectExtension.GetBoundingBox(keypoints);
                Vector2 center = rect.center;
                float size = Mathf.Min(rect.width, rect.height);
                rect.Set(center.x - size * 0.5f, center.y - size * 0.5f, size, size);

                FaceDetect.Result detection = new()
                {
                    score = score,
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
        }

        public const int KEYPOINT_COUNT = 468;
        private readonly float[,] output0 = new float[KEYPOINT_COUNT, 3]; // keypoint
        private readonly float[] output1 = new float[1]; // flag

        private readonly Result result;
        private Matrix4x4 cropMatrix;

        private readonly TensorToTexture debugInputTensorToTexture;

        public Vector2 FaceShift { get; set; } = new Vector2(0f, 0f);
        public Vector2 FaceScale { get; set; } = new Vector2(1.6f, 1.6f);

        public FaceDetect.Result Face { get; set; }
        public RenderTexture InputTexture => debugInputTensorToTexture.OutputTexture;

        public FaceMesh(string modelPath)
        {
            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AddGpuDelegate();
            Load(FileUtil.LoadFile(modelPath), interpreterOptions);
            AspectMode = AspectMode.Fill;

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

        public override void Dispose()
        {
            debugInputTensorToTexture.Dispose();
            base.Dispose();
        }


        protected override void PreProcess(Texture texture)
        {
            var face = Face;
            cropMatrix = RectTransformationCalculator.CalcMatrix(new()
            {
                rect = face.rect,
                rotationDegree = face.GetRotation() * Mathf.Rad2Deg,
                shift = FaceShift,
                scale = FaceScale,
                mirrorHorizontal = false,
                mirrorVertical = false,
            });

            var mtx = textureToTensor.GetAspectScaledMatrix(texture, AspectMode) * cropMatrix.inverse;

            var input = textureToTensor.Transform(texture, mtx);
            interpreter.SetInputTensorData(inputTensorIndex, input);

            debugInputTensorToTexture.Convert(input);
        }

        protected override void PostProcess()
        {
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



    }
}
