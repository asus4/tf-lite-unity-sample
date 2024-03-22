using System.Collections.Generic;
using System.Linq;
using Unity.Collections;
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
            [Range(0f, 1f)]
            public float threshold = 0.5f;
        }

        private Options options;
        private const int MAX_POINTS = 8;
        private readonly GraphicsBuffer pointsBuffer;
        private readonly TensorToTexture debugInputTensorToTexture;
        private readonly NativeArray<float> output0;
        private readonly TensorToTexture postProcessTensorToTexture;

        private static readonly int _InputPoints = Shader.PropertyToID("_InputPoints");
        private static readonly int _InputPointsCount = Shader.PropertyToID("_InputPointsCount");
        private static readonly int _Threshold = Shader.PropertyToID("_Threshold");


        public RenderTexture DebugInputTexture => debugInputTensorToTexture.OutputTexture;
        public RenderTexture OutputTexture => postProcessTensorToTexture.OutputTexture;

        public MagicTouch(string modelFile, Options options)
        {
            this.options = options;

            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AutoAddDelegate(options.delegateType, typeof(float));


            Load(FileUtil.LoadFile(modelFile), interpreterOptions);

            pointsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, MAX_POINTS, sizeof(float) * 2);
            debugInputTensorToTexture = new TensorToTexture(new()
            {
                compute = options.debugPreProcessCompute,
                width = width,
                height = height,
                channels = channels,
                inputType = interpreter.GetInputTensorInfo(0).type,
            });

            // Setup Output
            var outputInfo = interpreter.GetOutputTensorInfo(0);
            output0 = new NativeArray<float>(outputInfo.GetElementCount(), Allocator.Persistent);
            postProcessTensorToTexture = new TensorToTexture(new()
            {
                compute = options.postProcessCompute,
                width = outputInfo.shape[2],
                height = outputInfo.shape[1],
                channels = outputInfo.shape[3],
                inputType = outputInfo.type,
            });
        }

        public override void Dispose()
        {
            debugInputTensorToTexture.Dispose();
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
            interpreter.GetOutputTensorData(0, output0.AsSpan());
            options.postProcessCompute.SetFloat(_Threshold, options.threshold);
            postProcessTensorToTexture.Convert(output0);
        }
    }
}
