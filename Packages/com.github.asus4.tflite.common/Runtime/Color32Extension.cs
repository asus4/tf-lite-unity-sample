using UnityEngine;

namespace TensorFlowLite
{
    public static class Color32Extension
    {
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
