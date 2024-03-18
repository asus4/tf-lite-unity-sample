using System;
using System.Threading;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;

#if TFLITE_UNITASK_ENABLED
using Cysharp.Threading.Tasks;
#endif // TFLITE_UNITASK_ENABLED

using DataType = TensorFlowLite.Interpreter.DataType;

namespace TensorFlowLite
{
    /// <summary>
    /// Converts Texture to Tensor with arbitrary matrix transformation
    /// then return it as a NativeArray<byte> (NHWC layout)
    /// </summary>
    public abstract class TextureToNativeTensor : IDisposable
    {
        [Serializable]
        public class Options
        {
            public ComputeShader compute = null;
            public int kernel = 0;
            public int width = 0;
            public int height = 0;
            public int channels = 0;
            public DataType inputType = DataType.Float32;
        }

        protected static readonly Lazy<ComputeShader> DefaultComputeShaderFloat32 = new(()
            => Resources.Load<ComputeShader>("com.github.asus4.tflite.common/TextureToNativeTensorFloat32"));

        private static readonly int _InputTex = Shader.PropertyToID("_InputTex");
        private static readonly int _OutputTex = Shader.PropertyToID("_OutputTex");
        private static readonly int _OutputTensor = Shader.PropertyToID("_OutputTensor");
        private static readonly int _OutputSize = Shader.PropertyToID("_OutputSize");
        private static readonly int _TransformMatrix = Shader.PropertyToID("_TransformMatrix");

        private static readonly Matrix4x4 PopMatrix = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        private static readonly Matrix4x4 PushMatrix = Matrix4x4.Translate(new Vector3(-0.5f, -0.5f, 0));

        private readonly ComputeShader compute;
        private readonly int kernel;
        private readonly int width;
        private readonly int height;
        private readonly int channels;

        private readonly RenderTexture texture;
        private readonly GraphicsBuffer tensorBuffer;
        protected NativeArray<byte> tensor;

        public RenderTexture Texture => texture;
        public Matrix4x4 TransformMatrix { get; private set; } = Matrix4x4.identity;

        protected TextureToNativeTensor(int stride, Options options)
        {
            bool isSupported = SystemInfo.supportsAsyncGPUReadback && SystemInfo.supportsComputeShaders;
            if (!isSupported)
            {
                // Note: Async GPU Readback is supported on most platforms
                //       including OpenGL ES 3.0 since Unity 2021 LTS
                throw new NotSupportedException("AsyncGPUReadback and ComputeShader are required to use TextureToNativeTensor");
            }

            compute = options.compute != null
                ? options.compute
                : DefaultComputeShaderFloat32.Value;
            kernel = options.kernel;
            width = options.width;
            height = options.height;
            channels = options.channels;

            Assert.IsTrue(kernel >= 0, $"Kernel must be set");
            Assert.IsTrue(width > 0, $"Width must be greater than 0");
            Assert.IsTrue(height > 0, $"Height must be greater than 0");
            Assert.IsTrue(channels > 0 && channels <= 4, $"Channels must be 1 to 4");

            var desc = new RenderTextureDescriptor(width, height, RenderTextureFormat.ARGB32)
            {
                enableRandomWrite = true,
                useMipMap = false,
                depthBufferBits = 0,
            };
            texture = new RenderTexture(desc);
            texture.Create();

            int length = width * height * channels;
            tensorBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, length, stride);
            tensor = new NativeArray<byte>(length * stride, Allocator.Persistent);

            // Set constant values
            compute.SetInts(_OutputSize, width, height);
            compute.SetBuffer(kernel, _OutputTensor, tensorBuffer);
            compute.SetTexture(kernel, _OutputTex, texture, 0);
        }

        public virtual void Dispose()
        {
            texture.Release();
            UnityEngine.Object.Destroy(texture);
            tensorBuffer.Dispose();
        }

        public virtual NativeArray<byte> Transform(Texture input, in Matrix4x4 t)
        {
            TransformMatrix = t;
            compute.SetTexture(kernel, _InputTex, input, 0);
            compute.SetMatrix(_TransformMatrix, t);
            compute.Dispatch(kernel, Mathf.CeilToInt(width / 8f), Mathf.CeilToInt(height / 8f), 1);
            var request = AsyncGPUReadback.RequestIntoNativeArray(ref tensor, tensorBuffer);
            request.WaitForCompletion();
            return tensor;
        }

        public NativeArray<byte> Transform(Texture input, AspectMode aspectMode)
        {
            return Transform(input, GetAspectScaledMatrix(input, aspectMode));
        }

        public Matrix4x4 GetAspectScaledMatrix(Texture input, AspectMode aspectMode)
        {
            float srcAspect = (float)input.width / input.height;
            float dstAspect = (float)width / height;
            Vector2 scale = GetAspectScale(srcAspect, dstAspect, aspectMode);
            return PopMatrix * Matrix4x4.Scale(new Vector3(scale.x, scale.y, 1)) * PushMatrix;
        }

        public static Vector2 GetAspectScale(float srcAspect, float dstAspect, AspectMode mode)
        {
            bool isSrcWider = srcAspect > dstAspect;
            return (mode, isSrcWider) switch
            {
                (AspectMode.None, _) => new Vector2(1, 1),
                (AspectMode.Fit, true) => new Vector2(1, srcAspect / dstAspect),
                (AspectMode.Fit, false) => new Vector2(dstAspect / srcAspect, 1),
                (AspectMode.Fill, true) => new Vector2(dstAspect / srcAspect, 1),
                (AspectMode.Fill, false) => new Vector2(1, srcAspect / dstAspect),
                _ => throw new Exception("Unknown aspect mode"),
            };
        }

        /// <summary>
        /// Find the appropriate TextureToNativeTensor class for the given input type
        /// </summary>
        /// <param name="options">An options</param>
        /// <returns>An instance</returns>
        public static TextureToNativeTensor Create(Options options)
        {
            return options.inputType switch
            {
                DataType.Float32 => new TextureToNativeTensorFloat32(options),
                DataType.UInt8 => new TextureToNativeTensorUInt8(options),
                _ => throw new NotImplementedException(
                    $"input type {options.inputType} is not implemented yet. Create our own TextureToNativeTensor class and override it."),
            };
        }

        // Available when UniTask is installed
#if TFLITE_UNITASK_ENABLED

#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously
        public virtual async UniTask<NativeArray<byte>> TransformAsync(Texture input, Matrix4x4 t, CancellationToken cancellationToken)
        {
            TransformMatrix = t;
            compute.SetTexture(kernel, _InputTex, input, 0);
            compute.SetMatrix(_TransformMatrix, t);
            compute.Dispatch(kernel, Mathf.CeilToInt(width / 8f), Mathf.CeilToInt(height / 8f), 1);
            var request = AsyncGPUReadback.RequestIntoNativeArray(ref tensor, tensorBuffer);
            // Get this error 
            // AsyncGPUReadback - NativeArray does not have read/write access
            // https://forum.unity.com/threads/asyncgpureadback-requestintonativearray-causes-invalidoperationexception-on-nativearray.1011955/
            // await request.ToUniTask(cancellationToken: cancellationToken);
            request.WaitForCompletion();
            return tensor;
        }
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously

        public async UniTask<NativeArray<byte>> TransformAsync(Texture input, AspectMode aspectMode, CancellationToken cancellationToken)
        {
            return await TransformAsync(input, GetAspectScaledMatrix(input, aspectMode), cancellationToken);
        }
#endif // TFLITE_UNITASK_ENABLED

    }

    /// <summary>
    /// TextureToNativeTensor for float32 (float) input type
    /// </summary>
    public sealed class TextureToNativeTensorFloat32 : TextureToNativeTensor
    {
        public TextureToNativeTensorFloat32(Options options)
            : base(UnsafeUtility.SizeOf<float>(), options)
        { }
    }

    /// <summary>
    /// TextureToNativeTensor for uint8 (byte) input type
    /// 
    /// Note:
    /// Run compute shader with Float32 then convert to UInt8(byte) in C#
    /// Because ComputeBuffer doesn't support UInt8 type
    /// </summary>
    public sealed class TextureToNativeTensorUInt8 : TextureToNativeTensor
    {
        private NativeArray<byte> tensorUInt8;

        public TextureToNativeTensorUInt8(Options options)
            : base(UnsafeUtility.SizeOf<uint>(), options)
        {
            int length = options.width * options.height * options.channels;
            tensorUInt8 = new NativeArray<byte>(length, Allocator.Persistent);
            Assert.AreEqual(tensor.Length / 4, tensorUInt8.Length, $"Length {tensor.Length} != {tensorUInt8.Length}");
        }

        public override void Dispose()
        {
            base.Dispose();
            tensorUInt8.Dispose();
        }

        public override NativeArray<byte> Transform(Texture input, in Matrix4x4 t)
        {
            NativeArray<byte> tensor = base.Transform(input, t);
            // Reinterpret (byte * 4) as float
            NativeSlice<float> tensorF32 = tensor.Slice().SliceConvert<float>();

            // Cast Float32 to Uint8 using Burst
            var job = new CastFloat32toUInt8Job()
            {
                input = tensorF32,
                output = tensorUInt8,
            };
            job.Schedule().Complete();
            return tensorUInt8;
        }

#if TFLITE_UNITASK_ENABLED
        public override async UniTask<NativeArray<byte>> TransformAsync(Texture input, Matrix4x4 t, CancellationToken cancellationToken)
        {
            NativeArray<byte> tensor = await base.TransformAsync(input, t, cancellationToken);
            // Reinterpret (byte * 4) as float
            NativeSlice<float> tensorF32 = tensor.Slice().SliceConvert<float>();

            // Cast Float32 to Uint8 using Burst
            var job = new CastFloat32toUInt8Job()
            {
                input = tensorF32,
                output = tensorUInt8,
            };
            // wait for the job to complete async
            await job.Schedule();
            return tensorUInt8;
        }
#endif // TFLITE_UNITASK_ENABLED
    }

    /// <summary>
    /// Cast f32 to uint8 using Burst Job 
    /// </summary>
    [BurstCompile]
    internal struct CastFloat32toUInt8Job : IJob
    {
        [ReadOnly]
        public NativeSlice<float> input;

        [WriteOnly]
        public NativeArray<byte> output;

        public void Execute()
        {
            for (int i = 0; i < input.Length; i++)
            {
                // output[i] = (byte)Math.Clamp(input[i] * 255f, 0f, 255f);
                output[i] = (byte)(input[i] * 255f);
            }
        }
    }
}
