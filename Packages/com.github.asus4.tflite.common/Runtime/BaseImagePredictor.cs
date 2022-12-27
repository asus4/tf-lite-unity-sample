using System.Threading;
using UnityEngine;
#if TFLITE_UNITASK_ENABLED
using Cysharp.Threading.Tasks;
#endif // TFLITE_UNITASK_ENABLED

namespace TensorFlowLite
{
    public abstract class BaseImagePredictor<T> : System.IDisposable
        where T : struct
    {
        /// <summary>
        /// Accelerator options
        /// </summary>
        public enum Accelerator
        {
            NONE = 0,
            NNAPI = 1,
            GPU = 2,
            // HEXAGON = 3,
            XNNPACK = 4,
            // The EdgeTpu in Pixel devices.
            // EDGETPU = 5,
            // The Coral EdgeTpu Dev Board / USB accelerator.
            // EDGETPU_CORAL = 6,
        }

        protected readonly Interpreter interpreter;
        protected readonly int width;
        protected readonly int height;
        protected readonly int channels;
        protected readonly T[,,] inputTensor;
        protected readonly TextureToTensor tex2tensor;
        protected readonly TextureResizer resizer;
        protected TextureResizer.ResizeOptions resizeOptions;

        public Texture inputTex
        {
            get
            {
                return (tex2tensor.texture != null)
                    ? tex2tensor.texture as Texture
                    : resizer.texture as Texture;
            }
        }
        public Material transformMat => resizer.material;

        public TextureResizer.ResizeOptions ResizeOptions
        {
            get => resizeOptions;
            set => resizeOptions = value;
        }

        public BaseImagePredictor(string modelPath, InterpreterOptions options)
        {
            try
            {
                interpreter = new Interpreter(FileUtil.LoadFile(modelPath), options);
            }
            catch (System.Exception e)
            {
                interpreter?.Dispose();
                throw e;
            }

            interpreter.LogIOInfo();
            // Initialize inputs
            {
                var inputShape0 = interpreter.GetInputTensorInfo(0).shape;
                height = inputShape0[1];
                width = inputShape0[2];
                channels = inputShape0[3];
                inputTensor = new T[height, width, channels];

                int inputCount = interpreter.GetInputTensorCount();
                for (int i = 0; i < inputCount; i++)
                {
                    int[] shape = interpreter.GetInputTensorInfo(i).shape;
                    interpreter.ResizeInputTensor(i, shape);
                }
                interpreter.AllocateTensors();
            }

            tex2tensor = new TextureToTensor();
            resizer = new TextureResizer();
            resizeOptions = new TextureResizer.ResizeOptions()
            {
                aspectMode = AspectMode.Fill,
                rotationDegree = 0,
                mirrorHorizontal = false,
                mirrorVertical = false,
                width = width,
                height = height,
            };
        }

        public BaseImagePredictor(string modelPath, Accelerator accelerator)
            : this(modelPath, CreateOptions(accelerator))
        {
        }

        protected static InterpreterOptions CreateOptions(Accelerator accelerator)
        {
            var options = new InterpreterOptions();

            switch (accelerator)
            {
                case Accelerator.NONE:
                    options.threads = SystemInfo.processorCount;
                    break;
                case Accelerator.NNAPI:
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
                case Accelerator.GPU:
                    options.AddGpuDelegate();
                    break;
                case Accelerator.XNNPACK:
                    options.threads = SystemInfo.processorCount;
                    options.AddDelegate(XNNPackDelegate.DelegateForType(typeof(T)));
                    break;
                default:
                    options.Dispose();
                    throw new System.NotImplementedException();
            }
            return options;
        }

        public virtual void Dispose()
        {
            interpreter?.Dispose();
            tex2tensor?.Dispose();
            resizer?.Dispose();
        }

        public abstract void Invoke(Texture inputTex);

        protected void ToTensor(Texture inputTex, float[,,] inputs)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs);
        }

        protected void ToTensor(RenderTexture inputTex, float[,,] inputs, bool resize)
        {
            RenderTexture tex = resize ? resizer.Resize(inputTex, resizeOptions) : inputTex;
            tex2tensor.ToTensor(tex, inputs);
        }

        protected void ToTensor(Texture inputTex, float[,,] inputs, float offset, float scale)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs, offset, scale);
        }

        protected void ToTensor(Texture inputTex, sbyte[,,] inputs)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs);
        }

        protected void ToTensor(Texture inputTex, int[,,] inputs)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs);
        }

        // ToTensorAsync methods are only available when UniTask is installed via Unity Package Manager.
        // TODO: consider using native Task or Unity Coroutine
#if TFLITE_UNITASK_ENABLED
        protected async UniTask<bool> ToTensorAsync(Texture inputTex, float[,,] inputs, CancellationToken cancellationToken)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            await tex2tensor.ToTensorAsync(tex, inputs, cancellationToken);
            return true;
        }

        protected async UniTask<bool> ToTensorAsync(RenderTexture inputTex, float[,,] inputs, bool resize, CancellationToken cancellationToken)
        {
            RenderTexture tex = resize ? resizer.Resize(inputTex, resizeOptions) : inputTex;
            await tex2tensor.ToTensorAsync(tex, inputs, cancellationToken);
            return true;
        }

        protected async UniTask<bool> ToTensorAsync(Texture inputTex, sbyte[,,] inputs, CancellationToken cancellationToken)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            await tex2tensor.ToTensorAsync(tex, inputs, cancellationToken);
            return true;
        }

        protected async UniTask<bool> ToTensorAsync(Texture inputTex, int[,,] inputs, CancellationToken cancellationToken)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            await tex2tensor.ToTensorAsync(tex, inputs, cancellationToken);
            return true;
        }
#endif // TFLITE_UNITASK_ENABLED
    }
}
