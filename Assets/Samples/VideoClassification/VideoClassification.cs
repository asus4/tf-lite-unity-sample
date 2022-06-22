namespace TensorFlowLite
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using UnityEngine;
    using Accelerator = BaseImagePredictor<float>.Accelerator;

    /// <summary>
    /// Video Classification example from TensorFlow
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

        private readonly SignatureRunner runner;

        private const int IMAGE_TENSOR_INDEX = 37;

        private readonly Array[] states;

        public VideoClassification(Options modelOptions)
        {
            var options = new InterpreterOptions();
            switch (modelOptions.accelerator)
            {
                case Accelerator.NONE:
                    options.threads = SystemInfo.processorCount;
                    break;
                case Accelerator.NNAPI:
                    if (Application.platform == RuntimePlatform.Android)
                    {
                        options.useNNAPI = true;
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
                    options.AddDelegate(XNNPackDelegate.DelegateForType(typeof(float)));
                    break;
                default:
                    options.Dispose();
                    throw new NotImplementedException();
            }

            try
            {
                runner = new SignatureRunner(0, FileUtil.LoadFile(modelOptions.modelPath), options);
            }
            catch (Exception e)
            {
                runner?.Dispose();
                throw e;
            }

            runner.LogIOInfo();
        }

        public void Dispose()
        {
            runner?.Dispose();
        }

        public void Invoke(Texture inputTex)
        {
            Debug.Log($"Invoke : {inputTex.width}x{inputTex.height}");
        }
    }
}
