using UnityEngine;
using Unity.Mathematics;

namespace TensorFlowLite
{
    public static class ArrayExtension
    {
        public static int3 GetLengths(this int[,,] arr)
        {
            return new int3(
                arr.GetLength(0),
                arr.GetLength(1),
                arr.GetLength(2));
        }

        public static int3 GetLengths(this float[,,] arr)
        {
            return new int3(
                arr.GetLength(0),
                arr.GetLength(1),
                arr.GetLength(2));
        }
    }
}
