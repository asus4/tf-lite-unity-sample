using System;
using System.Linq;
using UnityEditor;
using NUnit.Framework;

namespace TensorFlowLite
{
    public class ArrayExtensionTest
    {
        [Test]
        public void ToIndexValueTuple()
        {
            var input = new int[] { 10, 11, 12, 13, 14 };
            var expected = new Tuple<int, int>[]
            {
                new Tuple<int, int>(0, 10),
                new Tuple<int, int>(1, 11),
                new Tuple<int, int>(2, 12),
                new Tuple<int, int>(3, 13),
                new Tuple<int, int>(4, 14),
            };
            var result = input.ToIndexValueTuple().ToArray();

            Assert.AreEqual(input.Length, result.Length);
            for (int i = 0; i < input.Length; i++)
            {
                AreEqual(expected[i], result[i]);
            }
        }

        private static void AreEqual<T, U>(Tuple<T, U> expected, Tuple<T, U> actual)
        {
            Assert.AreEqual(expected.Item1, actual.Item1);
            Assert.AreEqual(expected.Item2, actual.Item2);
        }
    }
}
