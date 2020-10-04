using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.Linq;

namespace PolyPerfect
{
    [CustomEditor(typeof(Animal_WanderScript))]
    [CanEditMultipleObjects]
    public class Animals_WanderScriptEditor : Common_WanderScriptEditor { }
}