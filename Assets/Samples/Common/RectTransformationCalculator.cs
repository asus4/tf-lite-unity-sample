using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// RectTransformationCalculator from MediaPipe
    /// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/rect_transformation_calculator.cc
    /// </summary>
    public class RectTransformationCalculator
    {
        public ref struct Options
        {
            /// <summary>
            /// The Normalized Rect (0, 0, 1, 1)
            /// </summary>
            public Rect rect;
            public float rotationDegree;
            public Vector2 shift;
            public Vector2 scale;

            // Custom options for WebCamTexture
            public float cameraRotationDegree;
            public bool mirrorHorizontal;
            public bool mirrorVertiacal;
        }

        private static readonly Matrix4x4 PUSH_MATRIX = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        private static readonly Matrix4x4 POP_MATRIX = Matrix4x4.Translate(new Vector3(-0.5f, -0.5f, 0));

        public static Matrix4x4 CalcMatrix(Options options)
        {

            Quaternion rotation = Quaternion.Euler(0, 0, options.rotationDegree);
            Vector2 size = Vector2.Scale(options.rect.size, options.scale);
            if (options.mirrorHorizontal)
            {
                size.x *= -1;
            }
            if (options.mirrorVertiacal)
            {
                size.y *= -1;
            }

            Vector2 shift = options.shift;
            // if (options.mirrorHorizontal)
            // {
            //     shift.x *= 1;
            // }
            // if (options.mirrorVertiacal)
            // {
            //     shift.y *= 1;
            // }

            // Calc center position
            Vector2 center = options.rect.center + new Vector2(-0.5f, -0.5f);
            center = (Vector2)(rotation * center);
            center += (shift * size);
            center /= size;

            Matrix4x4 trs = Matrix4x4.TRS(
                new Vector3(-center.x, -center.y, 0),
                rotation,
                new Vector3(1 / size.x, -1 / size.y, 1)
            );
            Matrix4x4 camTransform = Matrix4x4.Rotate(Quaternion.Euler(0, 0, options.cameraRotationDegree));
            // Matrix4x4 camTransform = Matrix4x4.Rotate(Quaternion.Euler(0, 0, 0));

            return PUSH_MATRIX * camTransform * trs * POP_MATRIX;
        }

        public static void DecodeToRectTransform(Matrix4x4 mtx, RectTransform t)
        {
            mtx = POP_MATRIX * mtx.inverse * PUSH_MATRIX;
            var position = mtx.ExtractPosition() + new Vector3(0.5f, 0.5f, 0);
            var rotation = mtx.ExtractRotation();
            var scale = mtx.ExtractScale();

            var size = t.sizeDelta;
            var min = new Vector3(-size.x / 2, -size.y / 2, 0);
            var max = new Vector3(size.x / 2, size.y / 2, 0);

            // Modify RectTransfrom setting
            t.pivot = t.anchorMin = t.anchorMax = new Vector2(0.5f, 0.5f);
            t.sizeDelta = ((RectTransform)t.parent).rect.size;

            // Set TRS
            t.localPosition = MathTF.Leap3Unclamped(min, max, position);
            t.localRotation = rotation;
            t.localScale = scale;
        }
    }

}
