namespace TensorFlowLite.MoveNet
{
    using UnityEngine;

    public class MoveNetDrawer : System.IDisposable
    {
        private readonly PrimitiveDraw draw;
        private readonly RectTransform view;
        private readonly Vector3[] rtCorners = new Vector3[4];

        public Color PoseColor { get; set; } = Color.green;
        public Color BoundingBoxColor { get; set; } = Color.blue;

        public MoveNetDrawer(Camera camera, RectTransform view)
        {
            this.view = view;
            draw = new PrimitiveDraw(camera, view.gameObject.layer);
        }

        public void Dispose()
        {
            draw.Dispose();
        }

        public void DrawPose(MoveNetPose pose, float threshold)
        {
            if (pose == null)
            {
                return;
            }

            draw.color = PoseColor;

            view.GetWorldCorners(rtCorners);
            Vector3 min = rtCorners[0];
            Vector3 max = rtCorners[2];

            var connections = PoseNet.Connections;
            int len = connections.GetLength(0);
            for (int i = 0; i < len; i++)
            {
                var a = pose[(int)connections[i, 0]];
                var b = pose[(int)connections[i, 1]];
                if (a.score >= threshold && b.score >= threshold)
                {
                    draw.Line3D(
                        MathTF.Lerp(min, max, new Vector3(a.x, 1f - a.y, 0)),
                        MathTF.Lerp(min, max, new Vector3(b.x, 1f - b.y, 0)),
                        1
                    );
                }
            }

            draw.Apply();
        }

        public void DrawPose(MoveNetPoseWithBoundingBox pose, float threshold)
        {
            if (pose == null)
            {
                return;
            }
            if (pose.score < threshold)
            {
                return;
            }
            DrawPose(pose as MoveNetPose, threshold);
            DrawBoundingBox(pose.boundingBox);
        }

        private void DrawBoundingBox(Rect rect)
        {
            Vector3 min = rtCorners[0];
            Vector3 max = rtCorners[2];

            draw.color = BoundingBoxColor;

            Vector3 p0 = MathTF.Lerp(min, max, new Vector3(rect.xMin, 1f - rect.yMin, 0));
            Vector3 p1 = MathTF.Lerp(min, max, new Vector3(rect.xMin, 1f - rect.yMax, 0));
            Vector3 p2 = MathTF.Lerp(min, max, new Vector3(rect.xMax, 1f - rect.yMax, 0));
            Vector3 p3 = MathTF.Lerp(min, max, new Vector3(rect.xMax, 1f - rect.yMin, 0));

            const float thickness = 0.5f;
            draw.Line3D(p0, p1, thickness);
            draw.Line3D(p1, p2, thickness);
            draw.Line3D(p2, p3, thickness);
            draw.Line3D(p3, p0, thickness);

            draw.Apply();
        }
    }
}
