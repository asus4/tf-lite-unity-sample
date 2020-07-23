using System.Linq;
using UnityEngine;

namespace TensorFlowLite
{
    public static class WebCamUtil
    {
        public struct PreferSpec
        {
            public bool isFrontFacing;
            public WebCamKind kind;

            public int GetScore(in WebCamDevice device)
            {
                int score = 0;
                if (device.isFrontFacing == isFrontFacing) score++;
                if (device.kind == kind) score++;
                return score;
            }
        }

        public static readonly PreferSpec DefaultPreferSpec = new PreferSpec()
        {
            isFrontFacing = false,
            kind = WebCamKind.WideAngle,
        };

        public static string FindName(PreferSpec spec = default(PreferSpec))
        {
            var devices = WebCamTexture.devices;
            if (Application.isMobilePlatform)
            {
                var prefers = devices.OrderByDescending(d => spec.GetScore(d));
                return prefers.First().name;
            }
            return devices.Last().name;
        }
    }
}
