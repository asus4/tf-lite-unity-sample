using UnityEngine;
using DataType = TensorFlowLite.Interpreter.DataType;

namespace TensorFlowLite
{
    public class HandLandmarkDetect : BaseVisionTask
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

        private readonly float[] output0 = new float[JOINT_COUNT * 2]; // keypoint
        private readonly float[] output1 = new float[1]; // hand flag
        private readonly Result result;
        private Matrix4x4 cropMatrix;
        private readonly TensorToTexture debugInputTensorToTexture;

        public Dimension Dim { get; private set; }
        public Vector2 PalmShift { get; set; } = new Vector2(0, 0.2f);
        public Vector2 PalmScale { get; set; } = new Vector2(2.8f, 2.8f);
        public Matrix4x4 CropMatrix => cropMatrix;

        public PalmDetect.Result Palm { get; set; }
        public RenderTexture InputTexture => debugInputTensorToTexture.OutputTexture;

        public HandLandmarkDetect(string modelPath)
        {
            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AddGpuDelegate();
            Load(FileUtil.LoadFile(modelPath), interpreterOptions);
            AspectMode = AspectMode.Fill;

            var out0info = interpreter.GetOutputTensorInfo(0);
            Dim = out0info.shape[1] switch
            {
                JOINT_COUNT * 2 => Dimension.TWO,
                JOINT_COUNT * 3 => Dimension.THREE,
                _ => throw new System.NotSupportedException(),
            };
            output0 = new float[out0info.shape[1]];

            result = new Result()
            {
                score = 0,
                joints = new Vector3[JOINT_COUNT],
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
            var palm = Palm;
            cropMatrix = RectTransformationCalculator.CalcMatrix(new()
            {
                rect = palm.rect,
                rotationDegree = palm.GetRotation() * Mathf.Rad2Deg,
                shift = PalmShift,
                scale = PalmScale,
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
    }
}
