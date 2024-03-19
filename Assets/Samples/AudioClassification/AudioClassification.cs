using System.Linq;

namespace TensorFlowLite
{
    public sealed class AudioClassification : System.IDisposable
    {
        private readonly Interpreter interpreter;
        private readonly float[] input;
        private readonly float[] output;

        public float[] Input => input;

        public AudioClassification(byte[] modelData)
        {
            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AutoAddDelegate(TfLiteDelegateType.XNNPACK, typeof(float));
            interpreter = new Interpreter(modelData, interpreterOptions);
            interpreter.LogIOInfo();

            // Allocate IO buffers
            input = new float[interpreter.GetInputTensorInfo(0).shape.Aggregate((x, y) => x * y)];
            output = new float[interpreter.GetOutputTensorInfo(0).shape.Aggregate((x, y) => x * y)];
            interpreter.AllocateTensors();
        }

        public void Dispose()
        {
            interpreter?.Dispose();
        }

        public void Run()
        {
            interpreter.SetInputTensorData(0, input);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output);
        }
    }
}
