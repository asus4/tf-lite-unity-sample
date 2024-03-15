namespace TensorFlowLite
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using UnityEngine;

    /// <summary>
    /// MoViNets: Video Classification example from TensorFlow
    /// https://www.tensorflow.org/lite/examples/video_classification/overview
    /// https://github.com/tensorflow/models/tree/master/official/projects/movinet
    /// </summary>
    public sealed class VideoClassification : IDisposable
    {
        [Serializable]
        public class Options
        {
            [FilePopup("*.tflite")]
            public string modelPath;
            public AspectMode aspectMode = AspectMode.Fill;
            public TfLiteDelegateType delegateType = TfLiteDelegateType.XNNPACK;
            public TextAsset labels;
        }

        public readonly struct Category : IComparable<Category>
        {
            public readonly int label;
            public readonly float score;

            public Category(int label, float score)
            {
                this.label = label;
                this.score = score;
            }

            public int CompareTo(Category other) => score > other.score ? -1 : 1;
        }

        private readonly string[] labels;
        private readonly SignatureRunner runner;

        private const string IMAGE_INPUT_NAME = "image";
        private const string LOGITS_OUTPUT_NAME = "logits";
        private const string SIGNATURE_KEY = "serving_default";
        private const int LABEL_COUNT = 600;
        private readonly Dictionary<string, Array> states = new Dictionary<string, Array>();
        private readonly float[] logitsTensor = new float[LABEL_COUNT];
        private readonly TextureToNativeTensor textureToTensor;
        private readonly AspectMode aspectMode;
        private readonly Category[] categories = new Category[LABEL_COUNT];

        public VideoClassification(Options options)
        {
            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AutoAddDelegate(options.delegateType, typeof(float));

            try
            {
                runner = new SignatureRunner(SIGNATURE_KEY, FileUtil.LoadFile(options.modelPath), interpreterOptions);
            }
            catch (Exception e)
            {
                runner?.Dispose();
                throw e;
            }

            runner.LogIOInfo();

            // Initialize inputs
            int width, height, channels;
            {
                var inputShape = runner.GetSignatureInputInfo(IMAGE_INPUT_NAME).shape;
                height = inputShape[2];
                width = inputShape[3];
                channels = inputShape[4];
            }
            foreach (string name in runner.InputSignatureNames)
            {
                if (name == IMAGE_INPUT_NAME)
                {
                    continue;
                }
                var info = runner.GetSignatureInputInfo(name);
                states.Add(name, ToArray(info));
            }

            textureToTensor = new TextureToNativeTensor(new TextureToNativeTensor.Options
            {
                compute = null,
                kernel = 0,
                width = width,
                height = height,
                channels = channels,
                inputType = typeof(float),
            });
            aspectMode = options.aspectMode;

            ResetStates();

            // Create labels
            labels = options.labels.text.Split('\n');
        }

        public void Dispose()
        {
            states.Clear();
            runner?.Dispose();
            textureToTensor.Dispose();
        }

        public void Invoke(Texture inputTex)
        {
            var input = textureToTensor.Transform(inputTex, aspectMode);

            runner.SetSignatureInputTensorData(IMAGE_INPUT_NAME, input);
            // Set inputs
            foreach (var kv in states)
            {
                runner.SetSignatureInputTensorData(kv.Key, kv.Value);
            }
            runner.Invoke();

            // Get outputs
            foreach (string name in runner.OutputSignatureNames)
            {
                if (name == LOGITS_OUTPUT_NAME)
                {
                    runner.GetSignatureOutputTensorData(name, logitsTensor);
                }
                else if (states.TryGetValue(name, out Array state))
                {
                    runner.GetSignatureOutputTensorData(name, state);
                }
                else
                {
                    Debug.LogError($"{name} is not found in output signature");
                }
            }
        }

        public IEnumerable<Category> GetResults()
        {
            var scores = logitsTensor.Softmax();
            int i = 0;
            foreach (var score in scores)
            {
                categories[i] = new Category(i, score);
                i++;
            }
            return categories.OrderByDescending(c => c.score);
        }

        public string GetLabel(int index)
        {
            return labels[index];
        }

        public void ResetStates()
        {
            foreach (var kv in states)
            {
                Array.Clear(kv.Value, 0, kv.Value.Length);
            }
        }

        private static Array ToArray(in Interpreter.TensorInfo info)
        {
            int length = info.shape.Aggregate(1, (acc, x) => acc * x);
            return info.type switch
            {
                Interpreter.DataType.Float32 => new float[length],
                Interpreter.DataType.Int32 => new int[length],
                _ => throw new NotImplementedException(),
            };
        }

    }
}
