using System.Linq;
using UnityEditor;
using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// A custom attribute that enables to select webcam name from the popup.
    /// </summary>
    [CustomPropertyDrawer(typeof(WebCamName))]
    public class WebCamNameDrawer : PropertyDrawer
    {
        string[] displayNames = null;
        int selectedIndex = -1;

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            if (property.propertyType != SerializedPropertyType.String)
            {
                Debug.LogError($"type: {property.propertyType} is not supported.");
                EditorGUI.LabelField(position, label.text, "Use WebcamName with string.");
                return;
            }

            if (displayNames == null)
            {
                // Init display names
                displayNames = WebCamTexture.devices.Select(device => device.name).ToArray();
            }

            if (selectedIndex < 0)
            {
                selectedIndex = FindSlectedIndex(displayNames, property.stringValue);
            }

            EditorGUI.BeginProperty(position, label, property);

            selectedIndex = EditorGUI.Popup(position, label.text, selectedIndex, displayNames);
            property.stringValue = displayNames[selectedIndex];

            EditorGUI.EndProperty();
        }

        private static int FindSlectedIndex(string[] displayNames, string value)
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
}
