namespace TensorFlowLite
{
    /// <summary>
    /// TensorFlow's delegate types
    /// https://www.tensorflow.org/lite/performance/delegates
    /// </summary>
    public enum TfLiteDelegateType
    {
        NONE = 0,
        NNAPI = 1,
        GPU = 2,
        // HEXAGON = 3,
        XNNPACK = 4,
        // The EdgeTpu in Pixel devices.
        // EDGETPU = 5,
        // The Coral EdgeTpu Dev Board / USB accelerator.
        // EDGETPU_CORAL = 6,
    }
}
