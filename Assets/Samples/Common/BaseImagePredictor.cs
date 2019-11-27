using System.Collections;
using System.Collections.Generic;
using System.IO;
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


        public BaseImagePredictor(string modelPath)
        {
            var options = new Interpreter.Options()
            {
                threads = 2,
                gpuDelegate = new MetalDelegate(new MetalDelegate.TFLGpuDelegateOptions()
                {
                    allow_precision_loss = false,
                    waitType = MetalDelegate.TFLGpuDelegateWaitType.Passive,
                })
            };
            interpreter = new Interpreter(File.ReadAllBytes(modelPath), options);

            interpreter.LogIOInfo();

            var idim0 = interpreter.GetInputTensorInfo(0).dimensions;
            height = idim0[1];
            width = idim0[2];
            channels = idim0[3];
            inputs = new T[height, width, channels];

            interpreter.ResizeInputTensor(0, idim0);

            tex2tensor = new TextureToTensor();
            resizeOptions = new TextureToTensor.ResizeOptions()
            {
                aspectMode = TextureToTensor.AspectMode.Fill,
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

        protected void ToTensor(Texture inputTex, sbyte[,,] inputs)
        {
            RenderTexture tex = tex2tensor.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs);
        }
    }
}