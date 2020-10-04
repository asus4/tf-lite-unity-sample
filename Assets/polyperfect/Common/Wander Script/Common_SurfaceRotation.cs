using UnityEngine;

namespace PolyPerfect
{
  public class Common_SurfaceRotation : MonoBehaviour
  {
    private string terrainLayer = "Terrain";
    private int layer;
    private bool rotate = true;
    private Quaternion targetRotation;
    private float rotationSpeed = 2f;

    private void Awake()
    {
      layer = LayerMask.GetMask(terrainLayer);
    }

    private void Start()
    {
      RaycastHit hit;
      Vector3 direction = transform.parent.TransformDirection(Vector3.down);

      if (Physics.Raycast(transform.parent.position, direction, out hit, 50f, layer))
      {
        float distance = hit.distance;
        Quaternion surfaceRotation = Quaternion.FromToRotation(Vector3.up, hit.normal);
        transform.rotation = surfaceRotation * transform.parent.rotation;
      }
    }

    private void Update()
    {
      if (!rotate)
        return;

      RaycastHit hit;
      Vector3 direction = transform.parent.TransformDirection(Vector3.down);

      if (Physics.Raycast(transform.parent.position, direction, out hit, 50f, layer))
      {
        float distance = hit.distance;
        Quaternion surfaceRotation = Quaternion.FromToRotation(Vector3.up, hit.normal);
        targetRotation = surfaceRotation * transform.parent.rotation;
      }

      transform.rotation = Quaternion.Lerp(transform.rotation, targetRotation, Time.deltaTime * rotationSpeed);
    }

    public void SetRotationSpeed(float speed)
    {
      if (speed > 0f)
        rotationSpeed = speed;
    }

    private void OnBecameVisible()
    {
      rotate = true;
    }

    private void OnBecameInvisible()
    {
      rotate = false;
    }
  }
}