using System;
using Unity.Mathematics;
using UnityEngine;

namespace TensorFlowLite
{
    public class BlazePoseDrawer : IDisposable
    {
        private readonly PrimitiveDraw draw;
        private readonly RectTransform container;
        private readonly Camera camera;
        // just cache for GetWorldCorners
        private readonly Vector3[] rtCorners = new Vector3[4];
        private readonly Vector4[] viewportLandmarks;

        public BlazePoseDrawer(Camera camera, int layer, RectTransform container)
        {
            draw = new PrimitiveDraw(camera, layer);
            this.container = container;
            this.camera = camera;
            viewportLandmarks = new Vector4[PoseLandmarkDetect.LandmarkCount];
        }

        public void Dispose()
        {
            draw.Dispose();
        }

        public void DrawPoseResult(PoseDetect.Result pose)
        {
            if (pose == null)
            {
                return;
            }
            container.GetWorldCorners(rtCorners);

            float3 min = rtCorners[0];
            float3 max = rtCorners[2];

            draw.color = Color.green;
            draw.Rect(MathTF.Lerp((Vector3)min, (Vector3)max, pose.rect.FlipY()), 0.02f, min.z);

            foreach (Vector2 kp in pose.keypoints)
            {
                draw.Point(math.lerp(min, max, new float3(kp.x, 1 - kp.y, 0)), 0.05f);
            }
            draw.Apply();
        }

        public void DrawCropMatrix(in Matrix4x4 matrix)
        {
            draw.color = Color.red;

            container.GetWorldCorners(rtCorners);
            Vector3 min = rtCorners[0];
            Vector3 max = rtCorners[2];

            Matrix4x4 mtx = matrix.inverse;
            Vector3 a = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(0, 0, 0)));
            Vector3 b = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(1, 0, 0)));
            Vector3 c = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(1, 1, 0)));
            Vector3 d = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(0, 1, 0)));

            draw.Quad(a, b, c, d, 0.02f);
            draw.Apply();
        }

        public void DrawLandmarkResult(PoseLandmarkDetect.Result result, float visibilityThreshold, float zOffset)
        {
            if (result == null)
            {
                return;
            }

            draw.color = Color.blue;

            Vector4[] landmarks = result.viewportLandmarks;
            // Update world joints
            for (int i = 0; i < landmarks.Length; i++)
            {
                Vector3 p = camera.ViewportToWorldPoint(landmarks[i]);
                viewportLandmarks[i] = new Vector4(p.x, p.y, p.z + zOffset, landmarks[i].w);
            }

            // Draw
            for (int i = 0; i < viewportLandmarks.Length; i++)
            {
                Vector4 p = viewportLandmarks[i];
                if (p.w > visibilityThreshold)
                {
                    draw.Cube(p, 0.2f);
                }
            }
            var connections = PoseLandmarkDetect.Connections;
            for (int i = 0; i < connections.Length; i += 2)
            {
                var a = viewportLandmarks[connections[i]];
                var b = viewportLandmarks[connections[i + 1]];
                if (a.w > visibilityThreshold || b.w > visibilityThreshold)
                {
                    draw.Line3D(a, b, 0.05f);
                }
            }
            draw.Apply();
        }

        public void DrawWorldLandmarks(PoseLandmarkDetect.Result result, float visibilityThreshold)
        {
            Vector4[] landmarks = result.viewportLandmarks;
            draw.color = Color.cyan;

            for (int i = 0; i < landmarks.Length; i++)
            {
                Vector4 p = landmarks[i];
                if (p.w > visibilityThreshold)
                {
                    draw.Cube(p, 0.02f);
                }
            }
            var connections = PoseLandmarkDetect.Connections;
            for (int i = 0; i < connections.Length; i += 2)
            {
                var a = landmarks[connections[i]];
                var b = landmarks[connections[i + 1]];
                if (a.w > visibilityThreshold || b.w > visibilityThreshold)
                {
                    draw.Line3D(a, b, 0.005f);
                }
            }

            draw.Apply();
        }

    }
}
