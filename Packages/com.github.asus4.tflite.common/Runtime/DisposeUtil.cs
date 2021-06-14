using UnityEngine;

namespace TensorFlowLite
{
    public static class DisposeUtil
    {
        public static void TryDispose(RenderTexture tex)
        {
            if (tex != null)
            {
                tex.Release();
                Object.Destroy(tex);
            }
        }

        public static void TryDispose(Texture2D tex)
        {
            if (tex != null)
            {
                Object.Destroy(tex);
            }
        }

        public static void TryDispose(Material mat)
        {
            if (mat != null)
            {
                Object.Destroy(mat);
            }
        }

        public static void TryDispose(ComputeBuffer buf)
        {
            if (buf != null)
            {
                buf.Dispose();
            }
        }
    }
}
