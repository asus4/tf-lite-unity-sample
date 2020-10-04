using UnityEngine;

namespace PolyPerfect
{
  public class Common_PlaySound : MonoBehaviour
  {
    [SerializeField]
    private AudioClip animalSound;
    [SerializeField]
    private AudioClip walking;
    [SerializeField]
    private AudioClip eating;
    [SerializeField]
    private AudioClip running;
    [SerializeField]
    private AudioClip attacking;
    [SerializeField]
    private AudioClip death;
    [SerializeField]
    private AudioClip sleeping;

    void AnimalSound()
    {
      if (animalSound)
      {
        Common_AudioManager.PlaySound(animalSound, transform.position);
      }
    }

    void Walking()
    {
      if (walking)
      {
        Common_AudioManager.PlaySound(walking, transform.position);
      }
    }

    void Eating()
    {
      if (eating)
      {
                Common_AudioManager.PlaySound(eating, transform.position);
      }
    }

    void Running()
    {
      if (running)
      {
                Common_AudioManager.PlaySound(running, transform.position);
      }
    }

    void Attacking()
    {
      if (attacking)
      {
                Common_AudioManager.PlaySound(attacking, transform.position);
      }
    }

    void Death()
    {
      if (death)
      {
                Common_AudioManager.PlaySound(death, transform.position);
      }
    }

    void Sleeping()
    {
      if (sleeping)
      {
                Common_AudioManager.PlaySound(sleeping, transform.position);
      }
    }
  }
}