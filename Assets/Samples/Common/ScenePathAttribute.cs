using System.Linq;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif // UNITY_EDITOR

namespace TensorFlowLite
{
    /// <summary>
    /// Attribute for scene path
    /// </summary>
    public class ScenePath : PropertyAttribute
    {
        public ScenePath()
        {
        }
    }

#if UNITY_EDITOR
    [CustomPropertyDrawer(typeof(ScenePath))]
    public class ScenePathDrawer : PropertyDrawer
    {
        private string[] scenePaths = null;

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            if (property.propertyType != SerializedPropertyType.String)
            {
                Debug.LogError($"type: {property.propertyType} is not supported.");
                EditorGUI.LabelField(position, label.text, "Use SceneName with string.");
                return;
            }

            // Init display names
            scenePaths ??= EditorBuildSettings.scenes
                .Select(scene => scene.path)
                .ToArray();

            EditorGUI.BeginProperty(position, label, property);
            int index = FindSelectedIndex(scenePaths, property.stringValue);
            int newIndex = EditorGUI.Popup(position, label.text, index, scenePaths);
            if (newIndex != index)
            {
                property.stringValue = scenePaths[newIndex];
            }
            EditorGUI.EndProperty();
        }

        private static int FindSelectedIndex(string[] scenePaths, string value)
        {
            for (int i = 0; i < scenePaths.Length; i++)
            {
                if (scenePaths[i] == value)
                {
                    return i;
                }
            }
            return 0;
        }
    }
#endif // UNITY_EDITOR
}
