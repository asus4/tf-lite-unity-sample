using System;
using System.Linq;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace TensorFlowLite
{
    public sealed class AudioClassification : IDisposable
    {
        public readonly struct Label : IComparable<Label>
        {
            public readonly int id;
            public readonly float score;

            public Label(int id, float score)
            {
                this.id = id;
                this.score = score;
            }

            public int CompareTo(Label other)
            {
                return other.score.CompareTo(score);
            }
        }

        private readonly Interpreter interpreter;
        private readonly float[] input;
        private readonly float[] output;
        private NativeArray<Label> labels;


        public float[] Input => input;
        public float[] Output => output;

        public AudioClassification(byte[] modelData)
        {
            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AutoAddDelegate(TfLiteDelegateType.XNNPACK, typeof(float));
            interpreter = new Interpreter(modelData, interpreterOptions);
            interpreter.LogIOInfo();

            // Allocate IO buffers
            input = new float[interpreter.GetInputTensorInfo(0).shape.Aggregate((x, y) => x * y)];
            output = new float[interpreter.GetOutputTensorInfo(0).shape.Aggregate((x, y) => x * y)];

            labels = new NativeArray<Label>(output.Length, Allocator.Persistent);
            interpreter.AllocateTensors();
        }

        public void Dispose()
        {
            interpreter?.Dispose();
            labels.Dispose();
        }

        public void Run()
        {
            interpreter.SetInputTensorData(0, input);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output);

            for (int i = 0; i < output.Length; i++)
            {
                labels[i] = new Label(i, output[i]);
            }

        }

        public NativeSlice<Label> GetTopLabels(int topK)
        {
            labels.Sort();
            return labels.Slice(0, topK);
        }

        [BurstCompile]
        internal struct OutPutToLabel : IJob
        {
            [ReadOnly]
            public NativeSlice<float> input;

            [WriteOnly]
            public NativeSlice<Label> output;

            public void Execute()
            {
                for (int i = 0; i < input.Length; i++)
                {
                    output[i] = new Label(i, input[i]);
                }
            }
        }
    }
}
