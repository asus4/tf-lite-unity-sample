using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// Utility to resize the texture
    /// </summary>
    public class TextureResizer : System.IDisposable
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
            public float rotationDegree;
            public bool mirrorHorizontal;
            public bool mirrorVertical;
            public AspectMode aspectMode;

            public ResizeOptions GetModifedForWebcam(WebCamTexture texture)
            {
                ResizeOptions modified = this; // copy
                int videoRotationAngle = texture.videoRotationAngle;
                if (rotationDegree - videoRotationAngle < 0)
                {
                    modified.rotationDegree = 360f + rotationDegree - videoRotationAngle;
                }
                else
                {
                    modified.rotationDegree = rotationDegree - videoRotationAngle;
                }

                bool needFlip90 = videoRotationAngle == 90 || videoRotationAngle == 270;
                if (needFlip90)
                {
                    modified.mirrorVertical = mirrorHorizontal;
                    modified.mirrorHorizontal = mirrorVertical;
                }

                if (texture.videoVerticallyMirrored)
                {
                    modified.mirrorVertical = !modified.mirrorVertical;
                }
                return modified;
            }
        }

        RenderTexture resizeTexture;
        Material _blitMaterial;

        static readonly int _VertTransform = Shader.PropertyToID("_VertTransform");
        static readonly int _UVRect = Shader.PropertyToID("_UVRect");

        public RenderTexture texture => resizeTexture;

        public Material material
        {
            get
            {
                if (_blitMaterial == null)
                {
                    _blitMaterial = new Material(Shader.Find("Hidden/TFLite/Resize"));
                }
                return _blitMaterial;
            }
        }

        public Vector4 UVRect
        {
            get => material.GetVector(_UVRect);
            set => material.SetVector(_UVRect, value);
        }

        public Matrix4x4 VertexTransfrom
        {
            get => material.GetMatrix(_VertTransform);
            set => material.SetMatrix(_VertTransform, value);
        }

        public TextureResizer()
        {

        }

        public void Dispose()
        {
            DisposeUtil.TryDispose(resizeTexture);
            DisposeUtil.TryDispose(_blitMaterial);
        }


        public RenderTexture Resize(Texture texture, ResizeOptions options)
        {
            // Set options
            if (texture is WebCamTexture)
            {
                options = options.GetModifedForWebcam((WebCamTexture)texture);
            }

            VertexTransfrom = GetVertTransform(options.rotationDegree, options.mirrorHorizontal, options.mirrorVertical);
            UVRect = GetTextureST(texture, options);
            return ApplyResize(texture, options.width, options.height, false);
        }

        public RenderTexture Resize(Texture texture,
            int width, int height, bool fillBackground,
             Matrix4x4 transform,
             Vector4 uvRect)
        {
            VertexTransfrom = transform;
            UVRect = uvRect;
            return ApplyResize(texture, width, height, fillBackground);
        }

        private RenderTexture ApplyResize(Texture texture, int width, int height, bool fillBackground)
        {
            if (resizeTexture == null
                || resizeTexture.width != width
                || resizeTexture.height != height)
            {
                DisposeUtil.TryDispose(resizeTexture);
                resizeTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
            }

            if (fillBackground)
            {
                // Fill with color 0,0,0,0
                Graphics.Blit(Texture2D.blackTexture, resizeTexture);
            }

            Graphics.Blit(texture, resizeTexture, material, 0);
            return resizeTexture;
        }

        private static Vector4 GetTextureST(float srcAspect, float dstAspect, AspectMode mode)
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

        public static Vector4 GetTextureST(Texture sourceTex, ResizeOptions options)
        {
            return GetTextureST(
                (float)sourceTex.width / (float)sourceTex.height, // src
                (float)options.width / (float)options.height, // dst
                options.aspectMode);
        }

        private static readonly Matrix4x4 PUSH_MATRIX = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        private static readonly Matrix4x4 POP_MATRIX = Matrix4x4.Translate(new Vector3(-0.5f, -0.5f, 0));
        private static Matrix4x4 GetVertTransform(float rotation, bool mirrorHorizontal, bool mirrorVertical)
        {
            Vector3 scale = new Vector3(
                mirrorHorizontal ? -1 : 1,
                mirrorVertical ? -1 : 1,
                1);
            Matrix4x4 trs = Matrix4x4.TRS(
                Vector3.zero,
                Quaternion.Euler(0, 0, rotation),
                scale
            );
            return PUSH_MATRIX * trs * POP_MATRIX;
        }

    }
}
