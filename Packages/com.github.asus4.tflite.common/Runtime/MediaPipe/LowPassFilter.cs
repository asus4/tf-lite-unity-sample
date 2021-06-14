
namespace TensorFlowLite
{
    /// <summary>
    /// Super simple low pass filter
    /// </summary>
    public class LowPassFilter
    {
        public float alpha;

        private float rawValue;
        private float storedValue;
        private bool isInitialized;

        public float Apply(float value)
        {
            float result;
            if (isInitialized)
            {
                result = alpha * value + (1.0f - alpha) * storedValue;
            }
            else
            {
                result = value;
                isInitialized = true;
            }
            rawValue = value;
            storedValue = result;
            return result;
        }

        public float Apply(float value, float alpha)
        {
            this.alpha = alpha;
            return Apply(value);
        }

    }
}
