using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// Utility to convert unity texture to Tensor
    /// </summary>
    public class TextureToTensor : System.IDisposable
    {
        public enum AspectMode
        {
            None,
            Fit,
            Fill,
        }

        public struct ResizeOptions
        {
            public int width;
            public int height;
            public bool flipX;
            public bool flipY;
            public AspectMode aspectMode;
        }

        RenderTexture resizeTexture;
        Material resizeMat;
        Texture2D fetchTexture;

        public Texture2D texture => fetchTexture;

        static readonly int _FlipX = Shader.PropertyToID("_FlipX");
        static readonly int _FlipY = Shader.PropertyToID("_FlipY");
        static readonly int _UVRect = Shader.PropertyToID("_UVRect");

        public TextureToTensor() { }

        public void Dispose()
        {
            TryDispose(resizeTexture);
            TryDispose(resizeMat);
            TryDispose(fetchTexture);
        }

        public RenderTexture Resize(Texture texture, ResizeOptions options)
        {
            if (resizeTexture == null
                        || resizeTexture.width != options.width
                        || resizeTexture.height != options.height)
            {
                TryDispose(resizeTexture);
                resizeTexture = new RenderTexture(options.width, options.height, 0, RenderTextureFormat.ARGB32);
            }
            if (resizeMat == null)
            {
                resizeMat = new Material(Shader.Find("Hidden/TFLite/Resize"));
            }

            // Set options
            resizeMat.SetInt(_FlipX, options.flipX ? 1 : 0);
            resizeMat.SetInt(_FlipY, options.flipY ? 1 : 0);
            resizeMat.SetVector(_UVRect, GetTextureST(
                (float)texture.width / (float)texture.height, // src
                (float)options.width / (float)options.height, // dst
                options.aspectMode));

            Graphics.Blit(texture, resizeTexture, resizeMat, 0);
            return resizeTexture;
        }

        public void ToTensor(RenderTexture texture, sbyte[,,] inputs)
        {
            // TODO: optimize this
            var pixels = FetchPixels(texture);
            int width = texture.width;

            for (int i = 0; i < pixels.Length; i++)
            {
                int y = i / width;
                int x = i % width;
                inputs[y, x, 0] = unchecked((sbyte)pixels[i].r);
                inputs[y, x, 1] = unchecked((sbyte)pixels[i].g);
                inputs[y, x, 2] = unchecked((sbyte)pixels[i].b);
            }
        }

        public void ToTensor01(RenderTexture texture, float[,,] inputs)
        {
            var pixels = FetchPixels(texture);
            int width = texture.width;
            const float scale = 255f;
            for (int i = 0; i < pixels.Length; i++)
            {
                int y = i / width;
                int x = i % width;
                inputs[y, x, 0] = (float)(pixels[i].r) / scale;
                inputs[y, x, 1] = (float)(pixels[i].g) / scale;
                inputs[y, x, 2] = (float)(pixels[i].b) / scale;
            }
        }

        public void ToTensor(RenderTexture texture, float[,,] inputs)
        {
            // TODO: optimize this
            var pixels = FetchPixels(texture);
            int width = texture.width;
            const float offset = 128f;
            for (int i = 0; i < pixels.Length; i++)
            {
                int y = i / width;
                int x = i % width;
                inputs[y, x, 0] = (unchecked((sbyte)pixels[i].r) - offset) / offset;
                inputs[y, x, 1] = (unchecked((sbyte)pixels[i].g) - offset) / offset;
                inputs[y, x, 2] = (unchecked((sbyte)pixels[i].b) - offset) / offset;
            }
        }

        Color32[] FetchPixels(RenderTexture texture)
        {
            if (fetchTexture == null || !IsSameSize(fetchTexture, texture))
            {
                fetchTexture = new Texture2D(texture.width, texture.height, TextureFormat.RGB24, 0, false);
            }
            var prevRT = RenderTexture.active;
            RenderTexture.active = texture;

            fetchTexture.ReadPixels(new Rect(0, 0, texture.width, texture.height), 0, 0);
            fetchTexture.Apply();

            RenderTexture.active = prevRT;

            return fetchTexture.GetPixels32();
        }

        static Vector4 GetTextureST(float srcAspect, float dstAspect, AspectMode mode)
        {
            switch (mode)
            {
                case AspectMode.None:
                    return new Vector4(1, 1, 0, 0);
                case AspectMode.Fit:
                    if (srcAspect > dstAspect)
                    {
                        float s = srcAspect / dstAspect;
                        return new Vector4(1, s, 0, (1 - s) / 2);
                    }
                    else
                    {
                        float s = dstAspect / srcAspect;
                        return new Vector4(s, 1, (1 - s) / 2, 0);
                    }
                case AspectMode.Fill:
                    if (srcAspect > dstAspect)
                    {
                        float s = dstAspect / srcAspect;
                        return new Vector4(s, 1, (1 - s) / 2, 0);
                    }
                    else
                    {
                        float s = srcAspect / dstAspect;
                        return new Vector4(1, s, 0, (1 - s) / 2);
                    }
            }
            throw new System.Exception("Unknown aspect mode");
        }

        public static Rect GetUVRect(float srcAspect, float dstAspect, AspectMode mode)
        {
            Vector4 texST = GetTextureST(srcAspect, dstAspect, mode);
            return new Rect(texST.z, texST.w, texST.x, texST.y);
        }

        static bool IsSameSize(Texture a, Texture b)
        {
            return a.width == b.width && a.height == b.height;
        }

        static void TryDispose(RenderTexture tex)
        {
            if (tex != null)
            {
                tex.Release();
                Object.Destroy(tex);
            }
        }

        static void TryDispose(Texture2D tex)
        {
            if (tex != null)
            {
                Object.Destroy(tex);
            }
        }

        static void TryDispose(Material mat)
        {
            if (mat == null)
            {
                Object.Destroy(mat);
            }
        }
    }
}
