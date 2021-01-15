using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace TensorFlowLite
{
    /// <summary>
    /// A simple drawing utility
    /// </summary>
    public sealed class PrimitiveDraw : System.IDisposable
    {
        #region Internal Class
        private class MeshBuffer : System.IDisposable
        {
            public Mesh mesh;
            public Matrix4x4[] buffer;
            public int index;

            public MeshBuffer(Mesh mesh, int initialSize = 256)
            {
                this.mesh = mesh;
                buffer = new Matrix4x4[initialSize];
                index = 0;
            }

            public void Dispose()
            {
                mesh = null;
                buffer = null;
            }

            public void Clear()
            {
                index = 0;
            }

            public void Add(in Matrix4x4 mtx)
            {
                buffer[index] = mtx;
                index++;
                if (index >= buffer.Length)
                {
                    var newBuffer = new Matrix4x4[buffer.Length * 2];
                    System.Array.Copy(buffer, newBuffer, buffer.Length);
                    buffer = newBuffer;
                    Debug.Log($"Allocate new buffer: {newBuffer.Length} mesh: {mesh.name}");
                }
            }
        }
        #endregion // Internal Class

        #region Private members

        private Material material;
        private MaterialPropertyBlock mpb;

        private MeshBuffer cube;
        private MeshBuffer quad;

        private Camera camera;
        private int layer;

        private static readonly int _Color = Shader.PropertyToID("_Color");

        #endregion // Private members

        #region Public

        public Color color
        {
            get => mpb.GetColor(_Color);
            set => mpb.SetColor(_Color, value);
        }

        public PrimitiveDraw(Camera camera = null, int layer = 0)
        {
            material = new Material(Shader.Find("Hidden/PrimitiveDraw"));
            material.hideFlags = HideFlags.HideAndDontSave;
            material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            material.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            material.SetInt("_ZWrite", 0);
            material.enableInstancing = true;

            mpb = new MaterialPropertyBlock();

            cube = new MeshBuffer(CreateMesh(PrimitiveType.Cube), 512);
            quad = new MeshBuffer(CreateMesh(PrimitiveType.Quad), 512);

            this.camera = camera ?? Camera.main;
            this.layer = layer;

            color = Color.green;
        }

        public void Dispose()
        {
            Object.Destroy(material);
            material = null;
            cube.Dispose();
            quad.Dispose();
        }

        public void Clear()
        {
            cube.Clear();
            quad.Clear();
        }

        public void Apply(bool drawEditor = true)
        {
            Draw(cube, drawEditor);
            Draw(quad, drawEditor);
            Clear();
        }

        public void Line(Vector3 start, Vector3 end, float thickness)
        {
            if (TryLine2DMatrix(start, end, thickness, out Matrix4x4 mtx))
            {
                quad.Add(mtx);
            }
        }

        public void Rect(Rect rect, float thickness, float z = 0)
        {
            if (rect.width <= 0 || rect.height <= 0) return;
            var p0 = new Vector3(rect.xMin, rect.yMin, z);
            var p1 = new Vector3(rect.xMax, rect.yMin, z);
            var p2 = new Vector3(rect.xMax, rect.yMax, z);
            var p3 = new Vector3(rect.xMin, rect.yMax, z);
            Matrix4x4 mtx;
            TryLine2DMatrix(p0, p1, thickness, out mtx);
            quad.Add(mtx);
            TryLine2DMatrix(p1, p2, thickness, out mtx);
            quad.Add(mtx);
            TryLine2DMatrix(p2, p3, thickness, out mtx);
            quad.Add(mtx);
            TryLine2DMatrix(p3, p0, thickness, out mtx);
            quad.Add(mtx);
        }

        public void Point(Vector3 p, float thickness)
        {
            var mtx = Matrix4x4.TRS(
                p,
                Quaternion.identity,
                new Vector3(thickness, thickness, thickness));
            quad.Add(mtx);
        }

        public void Line3D(Vector3 start, Vector3 end, float thickness)
        {
            if (TryLine3DMatrix(start, end, thickness, out Matrix4x4 mtx))
            {
                cube.Add(mtx);
            }
        }

        public void Cube(Vector3 center, float size)
        {
            if (size <= 0) return;
            var mtx = Matrix4x4.TRS(
                center,
                Quaternion.identity,
                new Vector3(size, size, size));
            cube.Add(mtx);
        }

        public void Quad(Vector3 a, Vector3 b, Vector3 c, Vector3 d, float thickness)
        {
            Line(a, b, thickness);
            Line(b, c, thickness);
            Line(c, d, thickness);
            Line(d, a, thickness);
        }

        #endregion // Public

        #region Private

        private void Draw(MeshBuffer mb, bool drawEditor)
        {
            if (mb.index <= 0) return;

            Graphics.DrawMeshInstanced(
                mb.mesh, 0, material, mb.buffer, mb.index,
                mpb, ShadowCastingMode.Off, false, layer, camera,
                LightProbeUsage.Off, null);

#if UNITY_EDITOR
            if (drawEditor && UnityEditor.SceneView.lastActiveSceneView != null)
            {
                var editorCamera = UnityEditor.SceneView.lastActiveSceneView.camera;
                Graphics.DrawMeshInstanced(
                    mb.mesh, 0, material, mb.buffer, mb.index,
                    mpb, ShadowCastingMode.Off, false, layer, editorCamera,
                    LightProbeUsage.Off, null);
            }
#endif // UNITY_EDITOR
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
        #endregion // Private Methods

    }

}
