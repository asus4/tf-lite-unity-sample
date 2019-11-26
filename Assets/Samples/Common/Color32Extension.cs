using UnityEngine;
using Unity.Mathematics;

namespace TensorFlowLite
{
    public static class Color32Extension
    {
        public static float4 ToARGB(this Color32 c)
        {
            return new float4(
                unchecked((sbyte)c.a) / 256f,
                unchecked((sbyte)c.r) / 256f,
                unchecked((sbyte)c.g) / 256f,
                unchecked((sbyte)c.b) / 256f
            );
        }

        public static float4 ToRGBA(this Color32 c)
        {
            return new float4(
                (float)unchecked((sbyte)c.r) / 255f,
                (float)unchecked((sbyte)c.g) / 255f,
                (float)unchecked((sbyte)c.b) / 255f,
                (float)unchecked((sbyte)c.a) / 255f
            );
        }

    }
}
