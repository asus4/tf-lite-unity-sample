using UnityEngine;
using Unity.Mathematics;

namespace TensorFlowLite
{
    public static class Color32Extension
    {
        public static float4 ToARGB(this Color32 c)
        {
            return new float4(
                unchecked((sbyte)c.a) / 255f,
                unchecked((sbyte)c.r) / 255f,
                unchecked((sbyte)c.g) / 255f,
                unchecked((sbyte)c.b) / 255f
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

        public static Color32 FromHex(uint c)
        {
            return new Color32()
            {
                a = unchecked((byte)(sbyte)((c & 0xFF000000) >> 24)),
                r = unchecked((byte)(sbyte)((c) & 0x00FF0000 >> 16)),
                g = unchecked((byte)(sbyte)((c & 0x0000FF00) >> 8)),
                b = unchecked((byte)(sbyte)((c & 0x000000FF))),
            };
        }
    }
}
