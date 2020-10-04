using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace PolyPerfect
{
  public class Common_AudioManager : MonoBehaviour
  {
    private static Common_AudioManager instance;
    [SerializeField]
    private bool muteSound;

    [SerializeField]
    private int objectPoolLength = 20;

    [SerializeField]
    private float soundDistance = 7f;

    [SerializeField]
    private bool logSounds = false;

    private List<AudioSource> pool = new List<AudioSource>();

    private void Awake()
    {
      instance = this;

      for (int i = 0; i < objectPoolLength; i++)
      {
        GameObject soundObject = new GameObject();
        soundObject.transform.SetParent(instance.transform);
        soundObject.name = "Sound Effect";
        AudioSource audioSource = soundObject.AddComponent<AudioSource>();
        audioSource.spatialBlend = 1f;
        audioSource.minDistance = instance.soundDistance;
        audioSource.gameObject.SetActive(false);
        pool.Add(audioSource);
      }
    }

    public static void PlaySound(AudioClip clip, Vector3 pos)
    {
      if (!instance)
      {
        Debug.LogError("No Audio Manager found in the scene.");
        return;
      }

      if(instance.muteSound)
      {
        return;
      }

      if (!clip)
      {
        Debug.LogError("Clip is null");
        return;
      }

      if (instance.logSounds)
      {
        Debug.Log("Playing Audio: " + clip.name);
      }

      for (int i = 0; i < instance.pool.Count; i++)
      {
        if (!instance.pool[i].gameObject.activeInHierarchy)
        {
          instance.pool[i].clip = clip;
          instance.pool[i].transform.position = pos;
          instance.pool[i].gameObject.SetActive(true);
          instance.pool[i].Play();
          instance.StartCoroutine(instance.ReturnToPool(instance.pool[i].gameObject, clip.length));
          return;
        }
      }

      GameObject soundObject = new GameObject();
      soundObject.transform.SetParent(instance.transform);
      soundObject.name = "Sound Effect";
      AudioSource audioSource = soundObject.AddComponent<AudioSource>();
      audioSource.spatialBlend = 1f;
      audioSource.minDistance = instance.soundDistance;
      instance.pool.Add(audioSource);
      audioSource.clip = clip;
      soundObject.transform.position = pos;
      audioSource.Play();
      instance.StartCoroutine(instance.ReturnToPool(soundObject, clip.length));
    }

    private IEnumerator ReturnToPool(GameObject obj, float delay)
    {
      yield return new WaitForSeconds(delay);
      obj.SetActive(false);
    }
  }
}