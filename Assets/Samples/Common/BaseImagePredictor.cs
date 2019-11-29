using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TensorFlowLite
{
    public abstract class BaseImagePredictor<T> : System.IDisposable
    {
        protected Interpreter interpreter;
        protected int width;
        protected int height;
        protected int channels;
        protected T[,,] inputs;
        protected TextureToTensor tex2tensor;
        protected TextureToTensor.ResizeOptions resizeOptions;

        public Texture2D inputTex => tex2tensor.texture;
        public Material transformMat => tex2tensor.material;

        public BaseImagePredictor(string modelPath, bool useGPU = true)
        {
            IGpuDelegate gpu = null;
            if (useGPU)
            {
#if UNITY_IOS || UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
                gpu = new MetalDelegate(new MetalDelegate.Options()
                {
                    allow_precision_loss = false,
                    waitType = MetalDelegate.WaitType.Passive,
                });
#endif
            }
            var options = new Interpreter.Options()
            {
                threads = 2,
                gpuDelegate = gpu,
            };
            interpreter = new Interpreter(FileUtil.LoadFile(modelPath), options);
            interpreter.LogIOInfo();
            InitInputs();

            tex2tensor = new TextureToTensor();
            resizeOptions = new TextureToTensor.ResizeOptions()
            {
                aspectMode = TextureToTensor.AspectMode.Fill,
                rotationDegree = 0,
                flipX = false,
                flipY = true,
                width = width,
                height = height,
            };
        }

        public virtual void Dispose()
        {
            interpreter?.Dispose();
            tex2tensor?.Dispose();
        }


        public abstract void Invoke(Texture inputTex);

        protected void ToTensor(Texture inputTex, float[,,] inputs)
        {
            RenderTexture tex = tex2tensor.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs);
        }

        protected void ToTensor(Texture inputTex, float[,,] inputs, float offset, float scale)
        {
            RenderTexture tex = tex2tensor.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs, offset, scale);
        }

        protected void ToTensor(Texture inputTex, sbyte[,,] inputs)
        {
            RenderTexture tex = tex2tensor.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs);
        }

        private void InitInputs()
        {
            var idim0 = interpreter.GetInputTensorInfo(0).dimensions;
            height = idim0[1];
            width = idim0[2];
            channels = idim0[3];
            inputs = new T[height, width, channels];

            int inputCount = interpreter.GetInputTensorCount();
            for (int i = 0; i < inputCount; i++)
            {
                int[] dim = interpreter.GetInputTensorInfo(i).dimensions;
                interpreter.ResizeInputTensor(i, dim);
            }
            interpreter.AllocateTensors();
        }
    }
}