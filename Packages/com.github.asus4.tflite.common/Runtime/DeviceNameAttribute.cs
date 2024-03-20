using UnityEngine;

namespace TensorFlowLite
{
    public class DeviceName : PropertyAttribute
    {
        public enum DeviceType
        {
            WebCam,
            Microphone
        }

        public DeviceType deviceType;

        public DeviceName(DeviceType deviceType)
        {
            this.deviceType = deviceType;
        }
    }

    /// <summary>
    /// Attribute for string to specify WebCam name in Editor
    /// </summary>
    public sealed class WebCamName : DeviceName
    {
        public WebCamName() : base(DeviceType.WebCam)
        {
        }
    }

    /// <summary>
    /// Attribute for string to specify Microphone name in Editor
    /// </summary>
    public sealed class MicrophoneName : DeviceName
    {
        public MicrophoneName() : base(DeviceType.Microphone)
        {
        }
    }
}
