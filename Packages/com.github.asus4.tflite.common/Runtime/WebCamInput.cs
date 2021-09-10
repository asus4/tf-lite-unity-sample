using UnityEngine;
using UnityEngine.Events;

namespace TensorFlowLite
{
    /// <summary>
    /// An wrapper for WebCamTexture that corrects texture rotation
    /// </summary>
    public sealed class WebCamInput : MonoBehaviour
    {
        [System.Serializable]
        public class TextureUpdateEvent : UnityEvent<Texture> { }

        [SerializeField, WebCamName] private string editorCameraName;
        [SerializeField] private WebCamKind preferKind = WebCamKind.WideAngle;
        [SerializeField] private bool isFrontFacing = false;
        [SerializeField] private int width = 1280;
        [SerializeField] private int height = 720;
        [SerializeField] private int fps = 60;
        public TextureUpdateEvent OnTextureUpdate = new TextureUpdateEvent();

        private TextureResizer resizer;
        private WebCamTexture webCamTexture;
        private WebCamDevice[] devices;

        private void Start()
        {
            resizer = new TextureResizer();
            devices = WebCamTexture.devices;
            string cameraName = Application.isEditor
                ? editorCameraName
                : WebCamUtil.FindName(preferKind, isFrontFacing);
            webCamTexture = new WebCamTexture(cameraName, width, height, fps);
            webCamTexture.Play();
        }

        private void OnDestroy()
        {
            resizer?.Dispose();
            webCamTexture?.Stop();
        }

        private void Update()
        {
            if (!webCamTexture.didUpdateThisFrame) return;

            var tex = NormalizeWebcam(webCamTexture, Screen.width, Screen.height, isFrontFacing);
            OnTextureUpdate.Invoke(tex);
        }

        private RenderTexture NormalizeWebcam(WebCamTexture texture, int width, int height, bool isFrontFacing)
        {
            int cameraWidth = texture.width;
            int cameraHeight = texture.height;
            bool isPortrait = IsPortrait(texture);
            if (isPortrait)
            {
                (cameraWidth, cameraHeight) = (cameraHeight, cameraWidth); // swap
            }

            float cameraAspect = (float)cameraWidth / (float)cameraHeight;
            float targetAspect = (float)width / (float)height;

            int w, h;
            if (cameraAspect > targetAspect)
            {
                w = Mathf.FloorToInt(cameraHeight * targetAspect);
                h = cameraHeight;
            }
            else
            {
                w = cameraWidth;
                h = Mathf.FloorToInt(cameraWidth / targetAspect);
            }

            Matrix4x4 mtx;
            Vector4 uvRect;
            int rotation = texture.videoRotationAngle;

            // Seems to be bug in the android. might be fixed in the future.
            if (Application.platform == RuntimePlatform.Android)
            {
                rotation = -rotation;
            }

            if (isPortrait)
            {
                mtx = TextureResizer.GetVertTransform(rotation, texture.videoVerticallyMirrored, isFrontFacing);
                uvRect = TextureResizer.GetTextureST(targetAspect, cameraAspect, AspectMode.Fill);
            }
            else
            {
                mtx = TextureResizer.GetVertTransform(rotation, isFrontFacing, texture.videoVerticallyMirrored);
                uvRect = TextureResizer.GetTextureST(cameraAspect, targetAspect, AspectMode.Fill);
            }

            // Debug.Log($"camera: rotation:{texture.videoRotationAngle} flip:{texture.videoVerticallyMirrored}");
            return resizer.Resize(texture, w, h, false, mtx, uvRect);
        }

        private static bool IsPortrait(WebCamTexture texture)
        {
            return texture.videoRotationAngle == 90 || texture.videoRotationAngle == 270;
        }
    }
}
