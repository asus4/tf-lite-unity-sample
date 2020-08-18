using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace TensorFlowLite
{
    public abstract class BaseImagePredictor<T> : System.IDisposable where T : struct
    {
        protected Interpreter interpreter;
        protected int width;
        protected int height;
        protected int channels;
        protected T[,,] input0;
        protected NativeArray<T> input;
        protected TextureToTensor tex2tensor;
        protected TextureResizer resizer;
        protected TextureResizer.ResizeOptions resizeOptions;

        public Texture2D inputTex => tex2tensor.texture;
        public Material transformMat => resizer.material;

        public TextureResizer.ResizeOptions ResizeOptions
        {
            get => resizeOptions;
            set => resizeOptions = value;
        }

        public BaseImagePredictor(string modelPath, bool useGPU = true)
        {
            var options = new InterpreterOptions()
            {
                threads = 2,
            };
            if (useGPU)
            {
                options.AddGpuDelegate();
            }

            interpreter = new Interpreter(FileUtil.LoadFile(modelPath), options);
            interpreter.LogIOInfo();
            InitInputs();

            tex2tensor = new TextureToTensor();
            resizer = new TextureResizer();
            resizeOptions = new TextureResizer.ResizeOptions()
            {
                aspectMode = TextureResizer.AspectMode.Fill,
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
            resizer?.Dispose();
            input.Dispose();
        }

        public abstract void Invoke(Texture inputTex);

        protected void ToTensor(Texture inputTex, ref NativeArray<T> inputs)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, ref inputs);
        }

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

        private void InitInputs()
        {
            var idim0 = interpreter.GetInputTensorInfo(0).shape;
            height = idim0[1];
            width = idim0[2];
            channels = idim0[3];
            input0 = new T[height, width, channels];
            input = new NativeArray<T>(height * width * channels, Allocator.Persistent, NativeArrayOptions.ClearMemory);

            int inputCount = interpreter.GetInputTensorCount();
            for (int i = 0; i < inputCount; i++)
            {
                int[] dim = interpreter.GetInputTensorInfo(i).shape;
                interpreter.ResizeInputTensor(i, dim);
            }
            interpreter.AllocateTensors();
        }
    }
}
