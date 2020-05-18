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
            public float rotationDegree;
            public bool flipX;
            public bool flipY;
            public AspectMode aspectMode;
        }

        RenderTexture resizeTexture;
        Material transfromMat;
        Texture2D fetchTexture;
        ComputeShader compute;
        ComputeBuffer tensorBuffer;

        public Texture2D texture => fetchTexture;
        public Material material => transfromMat;

        static readonly int _VertTransform = Shader.PropertyToID("_VertTransform");
        static readonly int _UVRect = Shader.PropertyToID("_UVRect");
        static readonly int InputTexture = Shader.PropertyToID("InputTexture");
        static readonly int OutputFloatTensor = Shader.PropertyToID("OutputFloatTensor");
        static readonly int TextureWidth = Shader.PropertyToID("TextureWidth");
        static readonly int TextureHeight = Shader.PropertyToID("TextureHeight");


        public TextureToTensor()
        {
            compute = Resources.Load<ComputeShader>("TextureToTensor");
        }

        public void Dispose()
        {
            TryDispose(resizeTexture);
            TryDispose(transfromMat);
            TryDispose(fetchTexture);
            TryDispose(tensorBuffer);
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
            if (transfromMat == null)
            {
                transfromMat = new Material(Shader.Find("Hidden/TFLite/Resize"));
            }

            // Set options
            float rotation = options.rotationDegree;
            if (texture is WebCamTexture)
            {
                var webcamTex = (WebCamTexture)texture;
                rotation += webcamTex.videoRotationAngle;
                if (webcamTex.videoVerticallyMirrored)
                {
                    options.flipX = !options.flipX;
                }
            }
            Matrix4x4 trs = GetVertTransform(rotation, options.flipX, options.flipY);
            transfromMat.SetMatrix(_VertTransform, trs);
            transfromMat.SetVector(_UVRect, GetTextureST(
                (float)texture.width / (float)texture.height, // src
                (float)options.width / (float)options.height, // dst
                options.aspectMode));

            Graphics.Blit(texture, resizeTexture, transfromMat, 0);
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
                inputs[y, x, 0] = ((sbyte)pixels[i].r);
                inputs[y, x, 1] = ((sbyte)pixels[i].g);
                inputs[y, x, 2] = ((sbyte)pixels[i].b);
            }
        }

        public void ToTensor(RenderTexture texture, float[,,] inputs)
        {
            if (texture.width % 8 != 0 || texture.height % 8 != 0)
            {
                ToTensorCPU(texture, inputs);
            }
            else
            {
                ToTensorGPU(texture, inputs);
            }
        }


        public void ToTensor(RenderTexture texture, float[,,] inputs, float offset, float scale)
        {
            // TODO: optimize this
            var pixels = FetchPixels(texture);
            int width = texture.width;
            for (int i = 0; i < pixels.Length; i++)
            {
                int y = i / width;
                int x = i % width;
                inputs[y, x, 0] = (pixels[i].r - offset) * scale;
                inputs[y, x, 1] = (pixels[i].g - offset) * scale;
                inputs[y, x, 2] = (pixels[i].b - offset) * scale;
            }
        }

        void ToTensorCPU(RenderTexture texture, float[,,] inputs)
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

        void ToTensorGPU(RenderTexture texture, float[,,] inputs)
        {
            int width = texture.width;
            int height = texture.height;

            if (tensorBuffer == null || tensorBuffer.count != width * height)
            {
                TryDispose(tensorBuffer);
                tensorBuffer = new ComputeBuffer(width * height, sizeof(float) * 3);
            }
            int kernel = compute.FindKernel("TextureToFloatTensor");

            compute.SetTexture(kernel, InputTexture, texture);
            compute.SetBuffer(kernel, OutputFloatTensor, tensorBuffer);
            compute.SetInt(TextureWidth, width);
            compute.SetInt(TextureHeight, height);

            compute.Dispatch(kernel, width / 8, height / 8, 1);

            tensorBuffer.GetData(inputs);
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
            if (mat != null)
            {
                Object.Destroy(mat);
            }
        }

        static void TryDispose(ComputeBuffer buf)
        {
            if (buf != null)
            {
                buf.Dispose();
            }
        }

        static readonly Matrix4x4 PUSH_MATRIX = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        static readonly Matrix4x4 POP_MATRIX = Matrix4x4.Translate(new Vector3(-0.5f, -0.5f, 0));
        static Matrix4x4 GetVertTransform(float rotation, bool invertX, bool invertY)
        {
            Vector3 scale = new Vector3(
                invertX ? -1 : 1,
                invertY ? -1 : 1,
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
