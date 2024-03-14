using System;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Assertions;

namespace TensorFlowLite
{
    /// <summary>
    /// Base class for task that takes a Texture as an input
    /// </summary>
    /// <typeparam name="T">A type of input tensor (float, sbyte etc.)</typeparam>
    public abstract class BaseVisionTask<T> : IDisposable
        where T : unmanaged
    {
        protected readonly Interpreter interpreter;
        protected readonly int width;
        protected readonly int height;
        protected readonly int channels;
        protected readonly TextureToNativeTensor<T> textureToTensor;

        public Texture InputTexture => textureToTensor.Texture;
        public AspectMode AspectMode { get; set; } = AspectMode.None;

        // Profilers
        static readonly ProfilerMarker preprocessPerfMarker = new($"{typeof(BaseVisionTask<T>).Name}.Preprocess");
        static readonly ProfilerMarker runPerfMarker = new($"{typeof(BaseVisionTask<T>).Name}.Session.Run");
        static readonly ProfilerMarker postprocessPerfMarker = new($"{typeof(BaseVisionTask<T>).Name}.Postprocess");

        /// <summary>
        /// Create in inference that has Image input
        /// </summary>
        /// <param name="model"></param>
        /// <param name="options"></param>
        public BaseVisionTask(byte[] model, InterpreterOptions options)
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
            // Initialize inputs
            {
                var inputShape0 = interpreter.GetInputTensorInfo(0).shape;
                Assert.AreEqual(4, inputShape0.Length);
                Assert.AreEqual(1, inputShape0[0], $"The batch size of the model must be 1. But got {inputShape0[0]}");
                height = inputShape0[1];
                width = inputShape0[2];
                channels = inputShape0[3];

                int inputCount = interpreter.GetInputTensorCount();
                for (int i = 0; i < inputCount; i++)
                {
                    int[] shape = interpreter.GetInputTensorInfo(i).shape;
                    interpreter.ResizeInputTensor(i, shape);
                }
                interpreter.AllocateTensors();
            }

            textureToTensor = new TextureToNativeTensor<T>(new TextureToNativeTensor<T>.Options
            {
                compute = null,
                kernel = 0,
                width = width,
                height = height,
                channels = channels,
            });
        }

        public virtual void Dispose()
        {
            interpreter?.Dispose();
            textureToTensor.Dispose();
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

        protected virtual void PreProcess(Texture texture)
        {
            // TODO: Support GPU binding
            var input = textureToTensor.Transform(texture, AspectMode);
            interpreter.SetInputTensorData(0, input);
        }

        protected virtual void PostProcess()
        {
            // Override this in subclass
        }

        /// <summary>
        /// Create InterpreterOptions from delegate type
        /// </summary>
        /// <param name="delegateType">A delegate type</param>
        /// <returns>An interpreter options</returns>
        protected static InterpreterOptions CreateOptions(TfLiteDelegateType delegateType)
        {
            var options = new InterpreterOptions();

            switch (delegateType)
            {
                case TfLiteDelegateType.NONE:
                    options.threads = SystemInfo.processorCount;
                    break;
                case TfLiteDelegateType.NNAPI:
                    if (Application.platform == RuntimePlatform.Android)
                    {
#if UNITY_ANDROID && !UNITY_EDITOR
                        // Create NNAPI delegate with default options
                        options.AddDelegate(new NNAPIDelegate());
#endif // UNITY_ANDROID && !UNITY_EDITOR
                    }
                    else
                    {
                        Debug.LogError("NNAPI is only supported on Android");
                    }
                    break;
                case TfLiteDelegateType.GPU:
                    options.AddGpuDelegate();
                    break;
                case TfLiteDelegateType.XNNPACK:
                    options.threads = SystemInfo.processorCount;
                    options.AddDelegate(XNNPackDelegate.DelegateForType(typeof(T)));
                    break;
                default:
                    options.Dispose();
                    throw new NotImplementedException();
            }
            return options;
        }
    }
}
