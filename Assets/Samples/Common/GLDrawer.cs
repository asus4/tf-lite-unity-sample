using System;
using UnityEngine;

namespace TensorFlowLite
{
    [RequireComponent(typeof(Camera))]
    public class GLDrawer : MonoBehaviour
    {
        public event Action OnDraw;

        Material lineMaterial;

        void OnEnable()
        {

            lineMaterial = new Material(Shader.Find("Hidden/Internal-Colored"));
            lineMaterial.hideFlags = HideFlags.HideAndDontSave;
            lineMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            lineMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            lineMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            lineMaterial.SetInt("_ZWrite", 0);
        }

        void OnDisable()
        {
            Destroy(lineMaterial);
        }

        void OnPostRender()
        {
            GL.PushMatrix();
            lineMaterial.SetPass(0);

            OnDraw();

            GL.PopMatrix();
        }

    }
}
