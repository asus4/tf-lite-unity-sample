namespace TensorFlowLite
{
    public class StylePredict : BaseVisionTask<float>
    {
        private readonly float[] output0;

        public StylePredict(string modelPath) :
            base(FileUtil.LoadFile(modelPath), CreateOptions(TfLiteDelegateType.GPU))
        {
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
