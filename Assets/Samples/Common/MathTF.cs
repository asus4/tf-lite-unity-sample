using System.Runtime.CompilerServices;
using UnityEngine;

namespace TensorFlowLite
{
    public static class MathTF
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(float x)
        {
            return (1.0f / (1.0f + Mathf.Exp(-x)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 Leap3(in Vector3 a, in Vector3 b, in Vector3 t)
        {
            return new Vector3(
                Mathf.Lerp(a.x, b.x, t.x),
                Mathf.Lerp(a.y, b.y, t.y),
                Mathf.Lerp(a.z, b.z, t.z)
            );
        }
    }
}
