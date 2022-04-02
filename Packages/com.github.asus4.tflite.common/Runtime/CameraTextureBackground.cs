namespace TensorFlowLite
{
    using UnityEngine;

    public class CameraTextureBackground : MonoBehaviour
    {
        [SerializeField]
        private bool useCustomMaterial = false;

        [SerializeField]
        private Material customMaterial = null;

        public Material Material { get; private set; }

        private static readonly int _UVRect = Shader.PropertyToID("_UVRect");

        protected virtual void Start()
        {
            Material = useCustomMaterial
                ? customMaterial
                : new Material(Shader.Find("Hidden/TFLite/Resize"));
            Material.SetMatrix("_VertTransform", Matrix4x4.identity);
        }

        protected virtual void OnDestroy()
        {
            if (!useCustomMaterial)
            {
                Destroy(Material);
            }
        }

        public virtual void SetTexture(Texture texture)
        {
            Material.mainTexture = texture;
            Vector4 uvRect = GetDisplayTransform(
                (float)texture.width / texture.height,
                (float)Screen.width / Screen.height);
            Material.SetVector(_UVRect, uvRect);
        }

        private static Vector4 GetDisplayTransform(float srcAspect, float dstAspect)
        {
            Vector2 scale;
            Vector2 offset;

            if (srcAspect > dstAspect)
            {
                float s = dstAspect / srcAspect;
                offset = new Vector2((1f - s) / 2f, 0);
                scale = new Vector2(s, 1);
            }
            else
            {
                float s = srcAspect / dstAspect;
                offset = new Vector3(0, (1f - s) / 2f);
                scale = new Vector3(1, s);
            }
            return new Vector4(scale.x, scale.y, offset.x, offset.y);
        }
    }
}
