namespace TensorFlowLite
{
    /// <summary>
    /// Specifies a mode for the model input image
    /// </summary>
    public enum AspectMode
    {
        /// <summary>
        /// Resizes the image without keeping the aspect ratio.
        /// </summary>
        None,
        /// <summary>
        /// Resizes the image to contain full area and padded black pixels.
        /// </summary>
        Fit,
        /// <summary>
        /// Trims the image to keep aspect ratio.
        /// </summary>
        Fill,
    }
}
