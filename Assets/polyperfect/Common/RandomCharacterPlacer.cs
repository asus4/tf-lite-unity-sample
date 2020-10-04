using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

namespace PolyPerfect
{
    [ExecuteInEditMode]
    public class RandomCharacterPlacer : MonoBehaviour
    {
        [SerializeField] float spawnSize;
        [SerializeField] int spawnAmmount;

        [SerializeField] GameObject[] characters;

        [ContextMenu("Spawn Characters")]
        void SpawnAnimals()
        {
            var parent = new GameObject("SpawnedCharacters");

            for (int i = 0; i < spawnAmmount; i++)
            {
                var value = Random.Range(0, characters.Length);

                Instantiate(characters[value], RandomNavmeshLocation(spawnSize), Quaternion.identity, parent.transform);
            }
        }

        public Vector3 RandomNavmeshLocation(float radius)
        {
            Vector3 randomDirection = Random.insideUnitSphere * radius;
            randomDirection += transform.position;
            NavMeshHit hit;
            Vector3 finalPosition = Vector3.zero;
            if (NavMesh.SamplePosition(randomDirection, out hit, radius, 1))
            {
                finalPosition = hit.position;
            }
            return finalPosition;
        }

        private void OnDrawGizmosSelected()
        {
            Gizmos.DrawWireSphere(transform.position, spawnSize);
        }
    }
}
