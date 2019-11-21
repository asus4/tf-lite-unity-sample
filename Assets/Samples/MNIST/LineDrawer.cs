using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.EventSystems;
using UnityEngine.UI;

[RequireComponent(typeof(RawImage))]
public class LineDrawer : MonoBehaviour, IDragHandler
{


    [System.Serializable]
    public class DrawEvent : UnityEvent<RenderTexture>
    {
    }


    [SerializeField] Vector2Int imageDimention = new Vector2Int(28, 28);
    [SerializeField] Color paintColor = Color.white;
    [SerializeField] RenderTextureFormat format = RenderTextureFormat.R8;

    [SerializeField] DrawEvent OnDraw = new DrawEvent();


    RawImage imageView = null;

    Mesh lineMesh;
    Material lineMaterial;
    Texture2D clearTexture;

    RenderTexture texture;

    void OnEnable()
    {
        imageView = GetComponent<RawImage>();

        lineMesh = new Mesh();
        lineMesh.MarkDynamic();
        lineMesh.vertices = new Vector3[2];
        lineMesh.SetIndices(new[] { 0, 1 }, MeshTopology.Lines, 0);
        lineMaterial = new Material(Shader.Find("Hidden/LineShader"));
        lineMaterial.SetColor("_Color", paintColor);

        texture = new RenderTexture(imageDimention.x, imageDimention.y, 0, format);
        texture.filterMode = FilterMode.Bilinear;
        imageView.texture = texture;

        clearTexture = Texture2D.blackTexture;

        var trigger = GetComponent<EventTrigger>();
    }

    void Start()
    {
        ClearTexture();
    }

    void OnDisable()
    {
        texture?.Release();
        Destroy(lineMesh);
        Destroy(lineMaterial);
    }

    public void ClearTexture()
    {
        Graphics.Blit(clearTexture, texture);
    }

    public void OnDrag(PointerEventData data)
    {
        data.Use();

        var area = data.pointerDrag.GetComponent<RectTransform>();
        var p0 = area.InverseTransformPoint(data.position - data.delta);
        var p1 = area.InverseTransformPoint(data.position);

        var scale = new Vector3(2 / area.rect.width, -2 / area.rect.height, 0);
        p0 = Vector3.Scale(p0, scale);
        p1 = Vector3.Scale(p1, scale);

        DrawLine(p0, p1);

        OnDraw.Invoke(texture);
    }

    void DrawLine(Vector3 p0, Vector3 p1)
    {
        var prevRT = RenderTexture.active;
        RenderTexture.active = texture;

        lineMesh.SetVertices(new List<Vector3>() { p0, p1 });
        lineMaterial.SetPass(0);
        Graphics.DrawMeshNow(lineMesh, Matrix4x4.identity);

        RenderTexture.active = prevRT;
    }
}

