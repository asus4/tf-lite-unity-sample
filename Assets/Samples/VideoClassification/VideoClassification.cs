namespace TensorFlowLite
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using UnityEngine;
    using Accelerator = BaseImagePredictor<float>.Accelerator;

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
            public Accelerator accelerator = Accelerator.XNNPACK;
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
        private readonly float[,,] inputTensor;
        private readonly float[] logitsTensor = new float[LABEL_COUNT];
        private readonly TextureToTensor tex2tensor;
        private readonly TextureResizer resizer;
        private TextureResizer.ResizeOptions resizeOptions;
        private readonly Category[] categories = new Category[LABEL_COUNT];

        public VideoClassification(Options options)
        {
            var interpreterOptions = new InterpreterOptions();
            switch (options.accelerator)
            {
                case Accelerator.NONE:
                    interpreterOptions.threads = SystemInfo.processorCount;
                    break;
                case Accelerator.NNAPI:
                    if (Application.platform == RuntimePlatform.Android)
                    {
                        interpreterOptions.useNNAPI = true;
                    }
                    else
                    {
                        Debug.LogError("NNAPI is only supported on Android");
                    }
                    break;
                case Accelerator.GPU:
                    interpreterOptions.AddGpuDelegate();
                    break;
                case Accelerator.XNNPACK:
                    interpreterOptions.threads = SystemInfo.processorCount;
                    interpreterOptions.AddDelegate(XNNPackDelegate.DelegateForType(typeof(float)));
                    break;
                default:
                    interpreterOptions.Dispose();
                    throw new NotImplementedException();
            }

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
            int width, height;
            {
                var inputShape = runner.GetSignatureInputInfo(IMAGE_INPUT_NAME).shape;
                height = inputShape[2];
                width = inputShape[3];
                int channels = inputShape[4];
                inputTensor = new float[height, width, channels];
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

            tex2tensor = new TextureToTensor();
            resizer = new TextureResizer();
            resizeOptions = new TextureResizer.ResizeOptions()
            {
                aspectMode = options.aspectMode,
                rotationDegree = 0,
                mirrorHorizontal = false,
                mirrorVertical = false,
                width = width,
                height = height,
            };

            ResetStates();

            // Create labels
            labels = options.labels.text.Split('\n');
        }

        public void Dispose()
        {
            states.Clear();
            runner?.Dispose();
            tex2tensor?.Dispose();
            resizer?.Dispose();
        }

        public void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, inputTensor);

            runner.SetSignatureInputTensorData(IMAGE_INPUT_NAME, inputTensor);
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

        private void ToTensor(Texture inputTex, float[,,] inputs)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs);
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
