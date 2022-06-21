namespace TensorFlowLite
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using UnityEngine;

    /// <summary>
    /// Video Classification example from TensorFlow
    /// https://www.tensorflow.org/lite/examples/video_classification/overview
    /// </summary>
    public class VideoClassification : BaseImagePredictor<sbyte>
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

        public VideoClassification(Options options)
            : base(options.modelPath, options.accelerator, IMAGE_TENSOR_INDEX)
        {
            resizeOptions.aspectMode = options.aspectMode;
        }


        public override void Dispose()
        {
            base.Dispose();
        }

        public override void Invoke(Texture inputTex)
        {
            // ToTensor(inputTex, input0);

            // interpreter.SetInputTensorData(0, input0);
            // interpreter.Invoke();
            // interpreter.GetOutputTensorData(0, outputs0);
        }
    }
}
