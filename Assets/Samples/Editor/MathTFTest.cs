using System;
using System.Linq;
using UnityEditor;
using NUnit.Framework;

namespace TensorFlowLite
{

    public class MathTFTest
    {
        [TestCase(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, },
                  new double[] { 0.011656231, 0.031684921, 0.086128544, 0.234121657, 0.636408647, })]
        [TestCase(new double[] { -1.0, -2.0, -3.0, -4.0, -5.0, },
                  new double[] { 0.636408647, 0.234121657, 0.086128544, 0.031684921, 0.011656231, })]
        public void SoftMaxDoubleTest(double[] input, double[] expected)
        {
            const double EPSILON = 0.00001;
            // const double EPSILON = double.Epsilon;
            ArrayEqual(input.Softmax().ToArray(), expected, EPSILON);
        }

        [TestCase(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, },
                  new float[] { 0.011656231f, 0.031684921f, 0.086128544f, 0.234121657f, 0.636408647f, })]
        [TestCase(new float[] { -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, },
                  new float[] { 0.636408647f, 0.234121657f, 0.086128544f, 0.031684921f, 0.011656231f, })]
        public void SoftMaxFloatTest(float[] input, float[] expected)
        {
            const float EPSILON = 0.00001f;
            // const float EPSILON = float.Epsilon;
            ArrayEqual(input.Softmax().ToArray(), expected, EPSILON);
        }

        private static void ArrayEqual(double[] actual, double[] expected, double epsilon)
        {
            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                double diff = Math.Abs(expected[i] - actual[i]);
                Assert.True(diff <= epsilon, $"expected: {expected[i]}, actual: {actual[i]}, diff: {diff}, epsilon: {epsilon}, diff/epsilon: {diff / epsilon}");
            }
        }

        private static void ArrayEqual(float[] actual, float[] expected, float epsilon)
        {
            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                float diff = Math.Abs(expected[i] - actual[i]);
                Assert.True(diff <= epsilon, $"expected: {expected[i]}, actual: {actual[i]}, diff: {diff}, epsilon: {epsilon}, diff/epsilon: {diff / epsilon}");
            }
        }

    }
}
