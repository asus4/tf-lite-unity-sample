using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif // UNITY_EDITOR

namespace TensorFlowLite
{
    public class SceneName : PropertyAttribute
    {
        public SceneName()
        {
        }
    }

#if UNITY_EDITOR
    [CustomPropertyDrawer(typeof(SceneName))]
    public class SceneNameDrawer : PropertyDrawer
    {
                string[] displayNames = null;

        int selectedIndex = -1;

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            if (property.propertyType != SerializedPropertyType.String)
            {
                Debug.LogError($"type: {property.propertyType} is not supported.");
                EditorGUI.LabelField(position, label.text, "Use SceneName with string.");
                return;
            }

            if (displayNames == null)
            {
                // Init display names
                displayNames = EditorBuildSettings.scenes
                    .Select(scene => scene.path)
                    .ToArray();
            }

            if (selectedIndex < 0)
            {
                selectedIndex = FindSelectedIndex(displayNames, property.stringValue);
            }

            EditorGUI.BeginProperty(position, label, property);

            selectedIndex = EditorGUI.Popup(position, label.text, selectedIndex, displayNames);
            property.stringValue = displayNames[selectedIndex];

            EditorGUI.EndProperty();
        }

        private static int FindSelectedIndex(string[] displayNames, string value)
        {
            for (int i = 0; i < displayNames.Length; i++)
            {
                if (displayNames[i] == value)
                {
                    return i;
                }
            }
            return 0;
        }
    }
#endif // UNITY_EDITOR
}
