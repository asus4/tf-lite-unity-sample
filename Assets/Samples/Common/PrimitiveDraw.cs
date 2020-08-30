using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// A simple drawing util
    /// </summary>
    public class PrimitiveDraw : System.IDisposable
    {
        private Material material;
        private Mesh cube;
        private Mesh quad;

        public Color color
        {
            get => material.color;
            set => material.color = value;
        }

        public PrimitiveDraw()
        {
            material = new Material(Shader.Find("Hidden/Internal-Colored"));
            material.hideFlags = HideFlags.HideAndDontSave;
            material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            material.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            material.SetInt("_ZWrite", 0);
            // 
            cube = CreateMesh(PrimitiveType.Cube);
            quad = CreateMesh(PrimitiveType.Quad);
        }

        public void Dispose()
        {
            Object.Destroy(material);
            material = null;
            cube = null;
            quad = null;
        }

        public void Line(Vector3 start, Vector3 end, float thickness)
        {
            if (TryLine2DMatrix(start, end, thickness, out Matrix4x4 mtx))
            {
                material.SetPass(0);
                Graphics.DrawMeshNow(quad, mtx);
            }
        }

        public void Rect(Rect rect, float thickness)
        {
            if (rect.width <= 0 || rect.height <= 0)
            {
                return;
            }
            var p0 = new Vector3(rect.xMin, rect.yMin, 0);
            var p1 = new Vector3(rect.xMax, rect.yMin, 0);
            var p2 = new Vector3(rect.xMax, rect.yMax, 0);
            var p3 = new Vector3(rect.xMin, rect.yMax, 0);
            material.SetPass(0);
            Matrix4x4 mtx;
            TryLine2DMatrix(p0, p1, thickness, out mtx);
            Graphics.DrawMeshNow(quad, mtx);
            TryLine2DMatrix(p1, p2, thickness, out mtx);
            Graphics.DrawMeshNow(quad, mtx);
            TryLine2DMatrix(p2, p3, thickness, out mtx);
            Graphics.DrawMeshNow(quad, mtx);
            TryLine2DMatrix(p3, p0, thickness, out mtx);
            Graphics.DrawMeshNow(quad, mtx);
        }

        public void Point(Vector3 p, float thickness)
        {
            material.SetPass(0);
            var mtx = Matrix4x4.TRS(
                p,
                Quaternion.Euler(0, 0, 0),
                new Vector3(thickness, thickness, thickness));
            Graphics.DrawMeshNow(quad, mtx);
        }

        public void Line3D(Vector3 start, Vector3 end, float thickness)
        {
            if (TryLine3DMatrix(start, end, thickness, out Matrix4x4 mtx))
            {
                material.SetPass(0);
                Graphics.DrawMeshNow(cube, mtx);
            }
        }

        public void Cube(Vector3 center, float size)
        {
            var mtx = Matrix4x4.TRS(
                center,
                Quaternion.identity,
                new Vector3(size, size, size));
            material.SetPass(0);
            Graphics.DrawMeshNow(cube, mtx);
        }

        private static bool TryLine2DMatrix(Vector3 start, Vector3 end, float thickness, out Matrix4x4 mtx)
        {
            var vec = end - start;
            var length = Vector3.Magnitude(vec);
            if (length < float.Epsilon)
            {
                mtx = Matrix4x4.identity;
                return false;
            }
            mtx = Matrix4x4.TRS(
                (end + start) / 2,
                Quaternion.Euler(0, 0, Mathf.Atan2(vec.y, vec.x) * Mathf.Rad2Deg),
                new Vector3(length, thickness, thickness));

            return true;
        }

        private static bool TryLine3DMatrix(Vector3 start, Vector3 end, float thickness, out Matrix4x4 mtx)
        {
            var vec = end - start;
            var length = Vector3.Magnitude(vec);
            if (length < float.Epsilon)
            {
                mtx = Matrix4x4.identity;
                return false;
            }
            mtx = Matrix4x4.TRS(
               (end + start) / 2,
               Quaternion.Euler(0, 0, Mathf.Atan2(vec.y, vec.x) * Mathf.Rad2Deg),
               new Vector3(length, thickness, thickness));

            return true;
        }

        private static Mesh CreateMesh(PrimitiveType type)
        {
            var go = GameObject.CreatePrimitive(type);
            Mesh mesh = go.GetComponent<MeshFilter>().sharedMesh;
            Object.Destroy(go);
            return mesh;
        }
    }

}
