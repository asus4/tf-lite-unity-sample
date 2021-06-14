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
            return devices.First().name;
        }

        private static readonly Matrix4x4 PUSH_MATRIX = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        private static readonly Matrix4x4 POP_MATRIX = Matrix4x4.Translate(new Vector3(-0.5f, -0.5f, 0));

        public static Matrix4x4 GetMatrix(float rotationDegree, bool mirrorHorizontal, bool mirrorVertical)
        {
            return
                PUSH_MATRIX
                * Matrix4x4.TRS(
                    new Vector3(0, 0, 0),
                    Quaternion.Euler(0, 0, rotationDegree),
                    new Vector3(
                        mirrorHorizontal ? -1 : 1,
                        mirrorVertical ? -1 : 1,
                        1
                    )
                )
                * POP_MATRIX;
        }
    }
}
