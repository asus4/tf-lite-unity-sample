using System;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Assertions;
using TensorInfo = TensorFlowLite.Interpreter.TensorInfo;

namespace TensorFlowLite
{
    /// <summary>
    /// Base class for task that takes a Texture as an input
    /// </summary>
    /// <typeparam name="T">A type of input tensor (float, sbyte etc.)</typeparam>
    public abstract class BaseVisionTask : IDisposable
    {
        protected Interpreter interpreter;
        protected int inputTensorIndex = 0;
        protected int width;
        protected int height;
        protected int channels;
        protected TextureToNativeTensor textureToTensor;

        public Texture InputTexture => textureToTensor.Texture;
        public AspectMode AspectMode { get; set; } = AspectMode.None;

        // Profilers
        static readonly ProfilerMarker preprocessPerfMarker = new($"{typeof(BaseVisionTask).Name}.Preprocess");
        static readonly ProfilerMarker runPerfMarker = new($"{typeof(BaseVisionTask).Name}.Session.Run");
        static readonly ProfilerMarker postprocessPerfMarker = new($"{typeof(BaseVisionTask).Name}.Postprocess");

        /// <summary>
        /// Load model from byte array
        /// </summary>
        /// <param name="model"></param>
        /// <param name="options"></param>
        public virtual void Load(byte[] model, InterpreterOptions options)
        {
            try
            {
                interpreter = new Interpreter(model, options);
            }
            catch (Exception e)
            {
                interpreter?.Dispose();
                throw e;
            }
#if UNITY_EDITOR
            interpreter.LogIOInfo();
#endif

            var inputTensorInfo = interpreter.GetInputTensorInfo(inputTensorIndex);
            InitializeInputsOutputs(inputTensorInfo);
            textureToTensor = CreateTextureToTensor(inputTensorInfo);
        }

        public virtual void Dispose()
        {
            interpreter?.Dispose();
            textureToTensor?.Dispose();
        }

        public virtual void Run(Texture texture)
        {
            // Pre process
            preprocessPerfMarker.Begin();
            PreProcess(texture);
            preprocessPerfMarker.End();

            // Run inference
            runPerfMarker.Begin();
            interpreter.Invoke();
            runPerfMarker.End();

            // Post process
            postprocessPerfMarker.Begin();
            PostProcess();
            postprocessPerfMarker.End();
        }

        /// <summary>
        /// Pre process the input texture
        /// Set all input tensors for the model
        /// </summary>
        /// <param name="texture">An input texture</param>
        protected virtual void PreProcess(Texture texture)
        {
            // TODO: Support GPU binding
            var input = textureToTensor.Transform(texture, AspectMode);
            interpreter.SetInputTensorData(inputTensorIndex, input);
        }

        /// <summary>
        /// Get the output tensors and do post process in subclass
        /// </summary>
        protected abstract void PostProcess();

        /// <summary>
        /// Default implementation of InitializeInputsOutputs
        /// Override this in subclass if needed
        /// </summary>
        protected virtual void InitializeInputsOutputs(TensorInfo inputTensorInfo)
        {
            int[] inputShape = inputTensorInfo.shape;
            Assert.AreEqual(4, inputShape.Length);
            Assert.AreEqual(1, inputShape[0], $"The batch size of the model must be 1. But got {inputShape[0]}");
            height = inputShape[1];
            width = inputShape[2];
            channels = inputShape[3];

            int inputCount = interpreter.GetInputTensorCount();
            for (int i = 0; i < inputCount; i++)
            {
                int[] shape = interpreter.GetInputTensorInfo(i).shape;
                interpreter.ResizeInputTensor(i, shape);
            }
            interpreter.AllocateTensors();
        }

        /// <summary>
        /// Create TextureToTensor for this model.
        /// Override this in subclass if needed
        /// </summary>
        /// <returns>A TextureToNativeTensor instance</returns>
        protected virtual TextureToNativeTensor CreateTextureToTensor(TensorInfo inputTensorInfo)
        {
            return TextureToNativeTensor.Create(new()
            {
                compute = null,
                kernel = 0,
                width = width,
                height = height,
                channels = channels,
                inputType = inputTensorInfo.type,
            });
        }
    }
}
