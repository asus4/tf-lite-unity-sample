namespace TensorFlowLite.MoveNet
{
    using UnityEngine;

    public class MoveNetDrawer : System.IDisposable
    {
        private readonly PrimitiveDraw draw;
        private readonly RectTransform view;
        private readonly Vector3[] rtCorners = new Vector3[4];

        public MoveNetDrawer(Camera camera, RectTransform view)
        {
            this.view = view;
            draw = new PrimitiveDraw(camera, view.gameObject.layer)
            {
                color = Color.green,
            };
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
    }
}
