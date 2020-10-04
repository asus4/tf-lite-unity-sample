using UnityEngine;
using System;

namespace PolyPerfect
{
  [Serializable]
  public class IdleState : AIState
  {
    public float minStateTime = 20f;
    public float maxStateTime = 40f;
    [Tooltip("Chance of it choosing this state, in comparion to other state weights.")]
    public int stateWeight = 20;
  }
}