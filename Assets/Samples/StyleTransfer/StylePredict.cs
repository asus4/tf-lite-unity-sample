namespace TensorFlowLite
{
    public class StylePredict : BaseVisionTask
    {
        private readonly float[] output0;

        public StylePredict(string modelPath)
        {
            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AddGpuDelegate();
            Load(FileUtil.LoadFile(modelPath), interpreterOptions);

            var outDim0 = interpreter.GetOutputTensorInfo(0).shape;
            output0 = new float[outDim0[3]]; // should be 100
        }

        protected override void PostProcess()
        {
            interpreter.GetOutputTensorData(0, output0);
        }

        public float[] GetStyleBottleneck()
        {
            return output0;
        }
    }
}
