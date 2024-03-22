using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// TensorFlow Lite Audio Classification
    /// https://www.tensorflow.org/lite/examples/audio_classification/overview
    /// https://www.kaggle.com/models/google/yamnet/frameworks/tensorFlow2/variations/yamnet/versions/1?tfhub-redirect=true
    /// </summary>
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
        private readonly NativeArray<float> output;
        private NativeArray<Label> labels;


        public float[] Input => input;

        public AudioClassification(byte[] modelData)
        {
            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AutoAddDelegate(TfLiteDelegateType.XNNPACK, typeof(float));
            interpreter = new Interpreter(modelData, interpreterOptions);
            interpreter.LogIOInfo();

            // Allocate IO buffers
            int inputLength = interpreter.GetInputTensorInfo(0).GetElementCount();
            input = new float[inputLength];
            int outputLength = interpreter.GetOutputTensorInfo(0).GetElementCount();
            output = new NativeArray<float>(outputLength, Allocator.Persistent);

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
            interpreter.GetOutputTensorData(0, output.AsSpan());

            var job = new OutPutToLabelJob()
            {
                input = output,
                output = labels,
            };
            job.Schedule().Complete();
        }

        public NativeSlice<Label> GetTopLabels(int topK)
        {
            return labels.Slice(0, topK);
        }

        [BurstCompile]
        internal struct OutPutToLabelJob : IJob
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
                output.Sort();
            }
        }
    }
}
