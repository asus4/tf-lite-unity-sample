namespace TensorFlowLite
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using UnityEngine;

    /// <summary>
    /// Video Classification example from TensorFlow
    /// https://www.tensorflow.org/lite/examples/video_classification/overview
    /// https://github.com/tensorflow/models/tree/master/official/projects/movinet
    /// </summary>
    public class VideoClassification : BaseImagePredictor<float>
    {
        [Serializable]
        public class Options
        {
            [FilePopup("*.tflite")]
            public string modelPath;
            public AspectMode aspectMode = AspectMode.Fill;
            public Accelerator accelerator = Accelerator.XNNPACK;
        }

        private const int IMAGE_TENSOR_INDEX = 37;

        private readonly Array[] states;

        public VideoClassification(Options options)
            : base(options.modelPath, options.accelerator, new int[] { 1, 172, 172, 3 })
        {
            resizeOptions.aspectMode = options.aspectMode;

            int inputCount = interpreter.GetInputTensorCount();
            states = new Array[inputCount];
            for (int i = 0; i < inputCount; i++)
            {
                Array arr = i == IMAGE_TENSOR_INDEX
                    ? inputTensor
                    : ToArray(interpreter.GetInputTensorInfo(i));
                states[i] = arr;
            }
        }

        public override void Dispose()
        {
            base.Dispose();
        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, inputTensor);

            int length = states.Length;
            for (int i = 0; i < length; i++)
            {
                interpreter.SetInputTensorData(i, states[i]);
            }
            interpreter.Invoke();
            for (int i = 0; i < length; i++)
            {
                if (i != IMAGE_TENSOR_INDEX)
                {
                    try
                    {
                        interpreter.GetOutputTensorData(i, states[i]);

                    }
                    catch (Exception e)
                    {
                        Debug.Log(e.Message);
                        Debug.Log($"index::::{i}");
                    }
                }
            }
        }

        private static Array ToArray(in Interpreter.TensorInfo info)
        {
            int length = info.shape.Aggregate(1, (acc, x) => acc * x);
            // Debug.Log($"{info} = {length}");
            return info.type switch
            {
                Interpreter.DataType.Float32 => new float[length],
                Interpreter.DataType.Int32 => new int[length],
                _ => throw new NotImplementedException(),
            };
        }
    }
}
