using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace PolyPerfect
{
    public class Common_StatsTable : EditorWindow
    {
        static List<AIStats> stats = new List<AIStats>();
        static public Dictionary<AIStats, int> selectionValue = new Dictionary<AIStats, int>();

        static string pathFolder = "";
        bool toggleName = true;
        bool toggleDominance = true;
        bool toggleAgression = true;
        bool toggleAttackSpeed = true;
        bool togglePower = true;
        bool toggleStamina = true;
        bool toggleStealthy = true;
        bool toggleToughness = true;
        bool toggleTeritorial = true;


        GUIStyle folderStyle = new GUIStyle();
        GUIStyle Explanation = new GUIStyle();
        GUIStyle toggleField = new GUIStyle();


        // Add menu named "My Window" to the Window menu
        [MenuItem("PolyPerfect/Stats Table")]
        static void Init()
        {
            // Get existing open window or if none, make a new one:
            Common_StatsTable window = (Common_StatsTable)EditorWindow.GetWindow(typeof(Common_StatsTable));
            window.Show();

            //If the window has been open before then clear the stats list and make a new one
            SortLists();
        }

        static void SortLists()
        {
            //Find all the stats in the project
            var Stats = (AIStats[])Resources.FindObjectsOfTypeAll(typeof(AIStats));

            //If the window has been open before then clear the stats list and make a new one
            stats.Clear();
            selectionValue.Clear();

            foreach (var item in Stats)
            {
                //Debug.Log(item.name);

                selectionValue.Add(item, -1);
                stats.Add(item);
            }
        }

        void OnDestroy()
        {
            stats.Clear();
            selectionValue.Clear();
        }

        void CreateNewStats()
        {
            var Stats = (AIStats[])Resources.FindObjectsOfTypeAll(typeof(AIStats));

            AIStats newStats = ScriptableObject.CreateInstance<AIStats>();

            if (AssetDatabase.GetMainAssetTypeAtPath(pathFolder + "/New  Stats.asset") != null)
            {
                AssetDatabase.CreateAsset(newStats, pathFolder + "/New  Stats" + Stats.Length.ToString() + ".asset");
            }

            else
            {
                AssetDatabase.CreateAsset(newStats, pathFolder + "/New  Stats.asset");
            }
        }


        void OnGUI()
        {
            folderStyle.normal.textColor = Color.black;
            Explanation.alignment = TextAnchor.MiddleCenter;

            pathFolder = "Assets";

            //Get the stats logo
            var mainTexture = Resources.Load<Texture2D>("StatsLogo");

            //Main Image    
            GUILayout.BeginHorizontal();
            if (GUILayout.Button(mainTexture))
            {
                Application.OpenURL("https://assetstore.unity.com/?q=Polyperfect&orderBy=0");
            }
            GUILayout.EndHorizontal();



            GUILayout.Label("See a side by side comparison of the  stats, use the boxes below to re order the list into the highest value of the category", Explanation);
            var filters = new List<string>();

            filters.Add(" Name");
            filters.Add("Dominance");
            filters.Add("Agression");
            filters.Add("AttackSpeed");
            filters.Add("Power");
            filters.Add("Stamina");
            filters.Add("Stealthy");
            filters.Add("Toughness");
            filters.Add("territorial");

            var buttonSize = (position.width / 9.5f);
            GUILayout.Space(20f);

            GUILayout.BeginHorizontal();

            if (GUILayout.Button(" Name", GUILayout.Width(buttonSize)))
            {
                //Re order by the s name
                ReOrderFloatList(0);
            }


            if (GUILayout.Button("Dominance", GUILayout.Width(buttonSize)))
            {
                ReOrderFloatList(1);
            }


            if (GUILayout.Button("Agression", GUILayout.Width(buttonSize)))
            {
                ReOrderFloatList(2);
            }

            if (GUILayout.Button("AttackSpeed", GUILayout.Width(buttonSize)))
            {
                ReOrderFloatList(3);
            }

            if (GUILayout.Button("Power", GUILayout.Width(buttonSize)))
            {
                ReOrderFloatList(4);
            }

            if (GUILayout.Button("Stamina", GUILayout.Width(buttonSize)))
            {
                ReOrderFloatList(5);
            }

            if (GUILayout.Button("Stealthy", GUILayout.Width(buttonSize)))
            {
                ReOrderFloatList(6);
            }

            if (GUILayout.Button("Toughness", GUILayout.Width(buttonSize)))
            {
                ReOrderFloatList(7);
            }


            if (GUILayout.Button("territorial", GUILayout.Width(buttonSize)))
            {
                ReOrderFloatList(8);
            }


            GUILayout.EndHorizontal();

            GUILayout.Space(20f);

            foreach (var item in stats)
            {
                BuildWindow(item);
            }

            if (GUILayout.Button("Add New Stats"))
            {
                if (AssetDatabase.IsValidFolder(pathFolder))
                {
                    CreateNewStats();
                }

                else
                {
                    Debug.Log("Please enter a valid path");
                }

                SortLists();
            }


            GUILayout.Space(20f);
        }

        void BuildWindow(AIStats Stats)
        {
            if (Stats == null)
            {
                stats.Remove(Stats);
            }

            toggleField.alignment = TextAnchor.MiddleCenter;

            Repaint();

            if (selectionValue.ContainsKey(Stats))
            {
                GUILayout.BeginHorizontal();

                var previousName = Stats.name;

                var newName = GUILayout.TextField(Stats.name, GUILayout.Width(position.width / 8.5f));

                Stats.name = newName;

                if (previousName != Stats.name)
                {
                    AssetDatabase.RenameAsset(AssetDatabase.GetAssetPath(Stats), Stats.name);
                    stats.Clear();
                    SortLists();
                }

                Stats.dominance = int.Parse(GUILayout.TextField(Stats.dominance.ToString(), GUILayout.Width(position.width / 9)));
                Stats.agression = Mathf.Clamp(float.Parse(GUILayout.TextField(Stats.agression.ToString(), GUILayout.Width(position.width / 9))), 0, 99);
                Stats.attackSpeed = float.Parse(GUILayout.TextField(Stats.attackSpeed.ToString(), GUILayout.Width(position.width / 9)));
                Stats.power = float.Parse(GUILayout.TextField(Stats.power.ToString(), GUILayout.Width(position.width / 9)));
                Stats.stamina = float.Parse(GUILayout.TextField(Stats.stamina.ToString(), GUILayout.Width(position.width / 9)));
                Stats.stealthy = GUILayout.Toggle(Stats.stealthy, Stats.stealthy.ToString(), GUILayout.Width(position.width / 9));
                Stats.toughness = float.Parse(GUILayout.TextField(Stats.toughness.ToString(), GUILayout.Width(position.width / 9)));
                Stats.territorial = GUILayout.Toggle(Stats.territorial, Stats.territorial.ToString(), toggleField, GUILayout.Width(position.width / 9));
                GUILayout.EndHorizontal();



            }
        }

        void ReOrderFloatList(int filterID)
        {
            switch (filterID)
            {
                case 0:

                    if (toggleName)
                    {
                        stats = (stats.OrderBy(p => p.name).Reverse()).ToList();
                        toggleName = !toggleName;
                    }

                    else
                    {
                        stats = stats.OrderBy(p => p.name).ToList();
                        toggleName = !toggleName;
                    }

                    break;

                case 1:
                    if (toggleDominance)
                    {
                        stats = (stats.OrderBy(p => p.dominance).Reverse()).ToList();
                        toggleDominance = !toggleDominance;
                    }

                    else
                    {
                        stats = stats.OrderBy(p => p.dominance).ToList();
                        toggleDominance = !toggleDominance;
                    }
                    break;

                case 2:

                    if (toggleAgression)
                    {
                        stats = (stats.OrderBy(p => p.agression).Reverse()).ToList();
                        toggleAgression = !toggleAgression;
                    }

                    else
                    {
                        stats = stats.OrderBy(p => p.agression).ToList();
                        toggleAgression = !toggleAgression;
                    }

                    break;

                case 3:

                    if (toggleAttackSpeed)
                    {
                        stats = (stats.OrderBy(p => p.attackSpeed).Reverse()).ToList();
                        toggleAttackSpeed = !toggleAttackSpeed;
                    }

                    else
                    {
                        stats = stats.OrderBy(p => p.attackSpeed).ToList();
                        toggleAttackSpeed = !toggleAttackSpeed;
                    }

                    break;

                case 4:

                    if (togglePower)
                    {
                        stats = (stats.OrderBy(p => p.power).Reverse()).ToList();
                        togglePower = !togglePower;
                    }

                    else
                    {
                        stats = stats.OrderBy(p => p.power).ToList();
                        togglePower = !togglePower;
                    }

                    break;

                case 5:

                    if (toggleStamina)
                    {
                        stats = (stats.OrderBy(p => p.stamina).Reverse()).ToList();
                        toggleStamina = !toggleStamina;
                    }

                    else
                    {
                        stats = stats.OrderBy(p => p.stamina).ToList();
                        toggleStamina = !toggleStamina;
                    }

                    break;

                case 6:

                    if (toggleStealthy)
                    {
                        stats = (stats.OrderBy(p => p.stealthy).Reverse()).ToList();
                        toggleStealthy = !toggleStealthy;
                    }

                    else
                    {
                        stats = stats.OrderBy(p => p.stealthy).ToList();
                        toggleStealthy = !toggleStealthy;
                    }

                    break;

                case 7:

                    if (toggleToughness)
                    {
                        stats = (stats.OrderBy(p => p.toughness).Reverse()).ToList();
                        toggleToughness = !toggleToughness;
                    }

                    else
                    {
                        stats = stats.OrderBy(p => p.toughness).ToList();
                        toggleToughness = !toggleToughness;
                    }

                    break;

                case 8:

                    if (toggleTeritorial)
                    {
                        stats = (stats.OrderBy(p => p.territorial).Reverse()).ToList();
                        toggleTeritorial = !toggleTeritorial;
                    }

                    else
                    {
                        stats = stats.OrderBy(p => p.territorial).ToList();
                        toggleTeritorial = !toggleTeritorial;
                    }

                    break;
            }
        }
    }
}