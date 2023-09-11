namespace TensorFlowLite
{
    using UnityEngine;
    using UnityEngine.Rendering;

    /// <summary>
    /// Add this component to a <c>Camera</c> to render the webcam texture (or any arbitrary texture) as a background.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Camera))]
    public class CameraTextureBackground : MonoBehaviour
    {
        [SerializeField]
        private bool useCustomMaterial = false;

        [SerializeField]
        private Material customMaterial = null;

        public Material Material { get; private set; }

        private static readonly int _UVRect = Shader.PropertyToID("_UVRect");

        #region Members for built-in render pipeline
        private static readonly Matrix4x4 _BackgroundOrthoProjection = Matrix4x4.Ortho(0f, 1f, 0f, 1f, -0.1f, 9.9f);
        private CommandBuffer _commandBuffer;
        private Mesh _backgroundMesh;
        #endregion // Members for built-in render pipeline

        protected virtual void Start()
        {
            Material = useCustomMaterial
                ? customMaterial
                : new Material(Shader.Find("Hidden/TFLite/Resize"));
            Material.SetMatrix("_VertTransform", Matrix4x4.identity);

            // Setup for built-in render pipeline
            if (GraphicsSettings.currentRenderPipeline == null)
            {
                SetupBuiltInRP();
            }
        }


        protected virtual void OnDestroy()
        {
            // Cleanup resources for built-in render pipeline
            if (_commandBuffer != null && TryGetComponent(out Camera camera))
            {
                camera.RemoveCommandBuffer(CameraEvent.BeforeForwardOpaque, _commandBuffer);
                _commandBuffer = null;

                Destroy(_backgroundMesh);
                _backgroundMesh = null;
            }

            // Cleanup resources for all
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

        private void SetupBuiltInRP()
        {
            if (!TryGetComponent(out Camera camera))
            {
                Debug.LogError($"{typeof(CameraTextureBackground).Name} requires a Camera component");
                return;
            }

            const string name = "Camera Texture Background Pass (Built-in)";
            var cmd = new CommandBuffer()
            {
                name = name,
            };

            cmd.BeginSample(name);

            cmd.SetInvertCulling(false);
            cmd.Blit(null, BuiltinRenderTextureType.CameraTarget, Material);

            cmd.EndSample(name);

            camera.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, cmd);
            _commandBuffer = cmd;
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
