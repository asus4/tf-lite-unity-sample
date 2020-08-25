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
        }

        public void Dispose()
        {
            Object.Destroy(material);
            material = null;
            Object.Destroy(cube);
            cube = null;
        }

        public void Line(Vector3 start, Vector3 end, float thickness)
        {
            var vec = end - start;
            var length = Vector3.Magnitude(vec);
            if (length < float.Epsilon)
            {
                return;
            }
            var mtx = Matrix4x4.TRS(
                (end + start) / 2,
                Quaternion.LookRotation(vec, Vector3.up),
                new Vector3(thickness, thickness, length));
            material.SetPass(0);
            Graphics.DrawMeshNow(cube, mtx);
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

        private static Mesh CreateMesh(PrimitiveType type)
        {
            var go = GameObject.CreatePrimitive(type);
            Mesh mesh = go.GetComponent<MeshFilter>().sharedMesh;
            Object.Destroy(go);
            return mesh;
        }
    }

}
