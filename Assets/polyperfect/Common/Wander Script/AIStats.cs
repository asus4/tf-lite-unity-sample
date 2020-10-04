using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace PolyPerfect
{
    [CreateAssetMenu(fileName = "New Wander Stats", menuName = "PolyPerfect/New Wander Stats", order = 1)]
    public class AIStats : ScriptableObject
    {
        [SerializeField, Tooltip("How dominent this is in the food chain, more agressive will attack less dominant.")]
        public int dominance = 1;

        [SerializeField, Tooltip("How many seconds this can run for before it gets tired.")]
        public float stamina = 10f;

        [SerializeField, Tooltip("How much this damage this does to another.")]
        public float power = 10f;

        [SerializeField, Tooltip("How much health this has.")]
        public float toughness = 5f;

        [SerializeField, Tooltip("Chance of this attacking another."), Range(0f, 100f)]
        public float agression = 0f;

        [SerializeField, Tooltip("How quickly this does damage to another (every 'attackSpeed' seconds will cause 'power' amount of damage).")]
        public float attackSpeed = 0.5f;

        [SerializeField, Tooltip("If true, this will attack other of the same specices.")]
        public bool territorial = false;

        [SerializeField, Tooltip("Stealthy can't be detected by others.")]
        public bool stealthy = false;
    }
}
