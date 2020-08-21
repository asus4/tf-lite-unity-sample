using UnityEngine;

namespace TensorFlowLite
{

    public class RectTransformationCalculator
    {
        private static readonly Matrix4x4 PUSH_MATRIX = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        private static readonly Matrix4x4 POP_MATRIX = Matrix4x4.Translate(new Vector3(-0.5f, -0.5f, 0));

        // https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/rect_transformation_calculator.cc
        public static Matrix4x4 CalcMatrix(Rect rect, float rotationDegree, Vector2 shift, Vector2 scale)
        {
            Quaternion rotation = Quaternion.Euler(0, 0, rotationDegree);

            // Calc scale
            Vector2 size = Vector2.Scale(rect.size, scale);

            // Calc center position
            Vector2 center;
            center = rect.center + new Vector2(-0.5f, -0.5f);
            center = (Vector2)(rotation * center);
            center += (shift * size);
            center /= size;

            Matrix4x4 trs = Matrix4x4.TRS(
                new Vector3(-center.x, -center.y, 0),
                rotation,
                new Vector3(1 / size.x, -1 / size.y, 1)
            );
            return PUSH_MATRIX * trs * POP_MATRIX;
        }
    }

}
