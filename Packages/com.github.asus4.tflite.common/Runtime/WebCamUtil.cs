using System.Linq;
using UnityEngine;

namespace TensorFlowLite
{
    public static class WebCamUtil
    {
        public struct PreferSpec
        {
            public readonly WebCamKind kind;
            public readonly bool isFrontFacing;

            public PreferSpec(WebCamKind kind, bool isFrontFacing)
            {
                this.kind = kind;
                this.isFrontFacing = isFrontFacing;
            }

            public int GetScore(in WebCamDevice device)
            {
                int score = 0;
                if (device.isFrontFacing == isFrontFacing) score++;
                if (device.kind == kind) score++;
                return score;
            }
        }

        public static readonly PreferSpec DefaultPreferSpec = new PreferSpec(WebCamKind.WideAngle, false);

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

        public static string FindName(WebCamKind kind, bool isFrontFacing)
        {
            return FindName(new PreferSpec(kind, isFrontFacing));
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
