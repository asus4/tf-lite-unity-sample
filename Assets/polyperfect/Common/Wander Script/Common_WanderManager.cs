using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace PolyPerfect
{
  public class Common_WanderManager : MonoBehaviour
  {
    [SerializeField]
    private bool peaceTime;
    public bool PeaceTime
    {
      get
      {
        return peaceTime;
      }
      set
      {
        SwitchPeaceTime(value);
      }
    }

    private static Common_WanderManager instance;
    public static Common_WanderManager Instance
    {
      get
      {
        return instance;
      }
    }

    private void Awake()
    {
      if (instance != null && instance != this)
      {
        Destroy(gameObject);
        return;
      }

      instance = this;
    }

    private void Start()
    {
      if (peaceTime)
      {
        Debug.Log("AnimalManager: Peacetime is enabled, all animals are non-agressive.");
        SwitchPeaceTime(true);
      }
    }

    public void SwitchPeaceTime(bool enabled)
    {
      if (enabled == peaceTime)
      {
        return;
      }

      peaceTime = enabled;

      Debug.Log(string.Format("AnimalManager: Peace time is now {0}.", enabled ? "On" : "Off"));
			foreach (Common_WanderScript animal in Common_WanderScript.AllAnimals)
      {
        animal.SetPeaceTime(enabled);
      }
    }

    public void Nuke()
    {
			foreach (Common_WanderScript animal in Common_WanderScript.AllAnimals)
      {
        animal.Die();
      }
    }
  }
}