using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using TensorInfo = TensorFlowLite.Interpreter.TensorInfo;

namespace TensorFlowLite
{
    /// <summary>
    /// MagicTouch model from MediaPipe's interactive segmentation task
    /// https://developers.google.com/mediapipe/solutions/vision/interactive_segmenter
    /// 
    /// Licensed under Apache License 2.0
    /// See model card for details
    /// https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MagicTouch.pdf
    /// </summary>
    public sealed class MagicTouch : BaseVisionTask
    {
        [System.Serializable]
        public class Options
        {
            public AspectMode aspectMode = AspectMode.Fit;
            public TfLiteDelegateType delegateType = TfLiteDelegateType.GPU;
            public ComputeShader preProcessCompute = null;
            public ComputeShader debugPreProcessCompute = null;
            public ComputeShader postProcessCompute = null;
        }

        private Options options;
        private const int MAX_POINTS = 8;
        private readonly GraphicsBuffer pointsBuffer;
        private readonly TensorToTexture debugInputTensorToTexture;

        private static readonly int _InputPoints = Shader.PropertyToID("_InputPoints");
        private static readonly int _InputPointsCount = Shader.PropertyToID("_InputPointsCount");

        public RenderTexture DebugInputTexture => debugInputTensorToTexture.OutputTexture;

        public MagicTouch(string modelFile, Options options)
        {
            this.options = options;

            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AutoAddDelegate(options.delegateType, typeof(float));

            pointsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, MAX_POINTS, sizeof(float) * 2);

            Load(FileUtil.LoadFile(modelFile), interpreterOptions);

            debugInputTensorToTexture = new TensorToTexture(new()
            {
                compute = options.debugPreProcessCompute,
                width = width,
                height = height,
                channels = channels,
                inputType = interpreter.GetInputTensorInfo(0).type,
            });
        }

        public override void Dispose()
        {
            pointsBuffer.Dispose();
            options = null;
            base.Dispose();
        }

        protected override TextureToNativeTensor CreateTextureToTensor(TensorInfo inputTensorInfo)
        {
            // Override pre-process compute shader
            return TextureToNativeTensor.Create(new()
            {
                compute = options.preProcessCompute,
                kernel = 0,
                width = width,
                height = height,
                channels = channels,
                inputType = inputTensorInfo.type,
            });
        }

        public void SetPoints(IEnumerable<Vector2> points)
        {
            int maxCount = pointsBuffer.count;
            int count = points.Count();
            if (count > maxCount)
            {
                Debug.LogWarning($"Too many points. Only the first {maxCount} points are used.");
                points = points.Take(maxCount);
            }
            pointsBuffer.SetData(points.ToArray());
            textureToTensor.compute.SetBuffer(0, _InputPoints, pointsBuffer);
            textureToTensor.compute.SetInt(_InputPointsCount, count);
        }

        protected override void PreProcess(Texture texture)
        {
            var input = textureToTensor.Transform(texture, AspectMode);
            interpreter.SetInputTensorData(inputTensorIndex, input);

            debugInputTensorToTexture.Convert(input);
        }

        protected override void PostProcess()
        {
        }
    }
}
