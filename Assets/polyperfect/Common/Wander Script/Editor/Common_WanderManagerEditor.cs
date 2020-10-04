using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

namespace PolyPerfect
{
    [CustomEditor(typeof(Common_WanderManager))]
    public class Common_WanderManagerEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            //Load a Texture (Assets/Resources/Textures/texture01.png)
            var mainTexture = Resources.Load<Texture2D>("ManagerLogo");
            GUILayout.BeginHorizontal();
            if (GUILayout.Button(mainTexture))
            {
                Application.OpenURL("https://assetstore.unity.com/?q=Polyperfect&orderBy=0");
            }
            GUILayout.EndHorizontal();

            Common_WanderManager Manager = (Common_WanderManager)target;

            if (!Application.isPlaying)
            {
                base.OnInspectorGUI();
                return;
            }

            GUILayout.Space(10);

            Manager.PeaceTime = EditorGUILayout.Toggle("Peace Time", Manager.PeaceTime);

            GUILayout.Space(5);

			if (GUILayout.Button("Kill 'Em All"))
            {
                Manager.Nuke();
            }
        }
    }
}