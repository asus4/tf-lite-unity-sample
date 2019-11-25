using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// Utility to convert unity texture to Tensor
    /// </summary>
    public class TextureToTensor : System.IDisposable
    {
        RenderTexture resizeTexture;
        Material resizeMat;
        Texture2D fetchTexture;

        public TextureToTensor() { }

        public void Dispose()
        {
            TryDispose(resizeTexture);
            TryDispose(resizeMat);
            TryDispose(fetchTexture);
        }

        public RenderTexture Resize(Texture texture, int width, int height)
        {
            if (resizeTexture == null
            || resizeTexture.width != width
            || resizeTexture.height != height)
            {
                TryDispose(resizeTexture);
                resizeTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
            }
            if (resizeMat == null)
            {
                resizeMat = new Material(Shader.Find("Hidden/TFLite/Flip"));
                resizeMat.SetInt("_FlipX", Application.isMobilePlatform ? 1 : 0);
                resizeMat.SetInt("_FlipY", 1);
            }

            Graphics.Blit(texture, resizeTexture, resizeMat, 0);
            return resizeTexture;
        }

        public void ToTensor(RenderTexture texture, sbyte[,,] inputs)
        {
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

        public void ToTensor(RenderTexture texture, float[,,] inputs)
        {
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

        static bool IsSameSize(Texture a, Texture b)
        {
            return a.width == b.width && a.height == b.height;
        }

        static void TryDispose(RenderTexture tex)
        {
            if (tex != null)
            {
                // Debug.Log($"RenderTex Dispose: {tex.width} x {tex.height}");
                tex.Release();
                Object.Destroy(tex);
            }
        }

        static void TryDispose(Texture2D tex)
        {
            if (tex != null)
            {
                // Debug.Log($"Texture2D Dispose: {tex.width} x {tex.height}");
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
