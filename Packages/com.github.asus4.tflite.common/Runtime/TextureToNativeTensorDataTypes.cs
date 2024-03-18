using System.Threading;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Assertions;

#if TFLITE_UNITASK_ENABLED
using Cysharp.Threading.Tasks;
#endif // TFLITE_UNITASK_ENABLED

namespace TensorFlowLite
{
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

        [BurstCompile]
        struct CastFloat32toUInt8Job : IJob
        {
            [ReadOnly]
            public NativeSlice<float> input;

            [WriteOnly]
            public NativeArray<byte> output;

            public void Execute()
            {
                for (int i = 0; i < input.Length; i++)
                {
                    output[i] = (byte)(input[i] * 255f);
                }
            }
        }
    }

    /// <summary>
    /// TextureToNativeTensor for int32 (int) input type
    /// </summary>
    public sealed class TextureToNativeTensorInt32 : TextureToNativeTensor
    {
        private NativeArray<byte> tensorInt32;

        public TextureToNativeTensorInt32(Options options)
            : base(UnsafeUtility.SizeOf<uint>(), options)
        {
            int length = options.width * options.height * options.channels;
            int stride = UnsafeUtility.SizeOf<int>();
            tensorInt32 = new NativeArray<byte>(length * stride, Allocator.Persistent);
            Assert.AreEqual(tensor.Length, tensorInt32.Length, $"Length {tensor.Length} != {tensorInt32.Length}");
        }

        public override void Dispose()
        {
            base.Dispose();
            tensorInt32.Dispose();
        }

        public override NativeArray<byte> Transform(Texture input, in Matrix4x4 t)
        {
            NativeArray<byte> tensor = base.Transform(input, t);
            // Reinterpret (byte * 4) as float
            NativeSlice<float> sliceF32 = tensor.Slice().SliceConvert<float>();
            // Reinterpret (byte * 4) as int
            NativeSlice<int> sliceI32 = tensorInt32.Slice().SliceConvert<int>();

            // Cast Float32 to Int32 using Burst
            var job = new CastFloat32toInt32Job()
            {
                input = sliceF32,
                output = sliceI32,
            };
            job.Schedule().Complete();
            return tensorInt32;
        }

#if TFLITE_UNITASK_ENABLED
        public override async UniTask<NativeArray<byte>> TransformAsync(Texture input, Matrix4x4 t, CancellationToken cancellationToken)
        {
            NativeArray<byte> tensor = await base.TransformAsync(input, t, cancellationToken);
            // Reinterpret (byte * 4) as float
            NativeSlice<float> sliceF32 = tensor.Slice().SliceConvert<float>();
            // Reinterpret (byte * 4) as int
            NativeSlice<int> sliceI32 = tensorInt32.Slice().SliceConvert<int>();

            // Cast Float32 to Uint8 using Burst
            var job = new CastFloat32toInt32Job()
            {
                input = sliceF32,
                output = sliceI32,
            };
            // wait for the job to complete async
            await job.Schedule();
            return tensorInt32;
        }
#endif // TFLITE_UNITASK_ENABLED

        /// <summary>
        /// Cast f32 to uint8 using Burst Job 
        /// </summary>
        [BurstCompile]
        internal struct CastFloat32toInt32Job : IJob
        {
            [ReadOnly]
            public NativeSlice<float> input;

            [WriteOnly]
            public NativeSlice<int> output;

            public void Execute()
            {
                for (int i = 0; i < input.Length; i++)
                {
                    output[i] = (int)(input[i] * 255f);
                }
            }
        }
    }
}
