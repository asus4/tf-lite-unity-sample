using System.Collections;
using TensorFlowLite;
using UnityEngine;

public class IndexSample : MonoBehaviour
{
    [System.Serializable]
    public class SceneInfo
    {
        public string displayName;
        [ScenePath]
        public string path;
    }

    [SerializeField]
    private RectTransform parent = default;

    [SerializeField]
    private GoToSceneButton buttonPrefab = default;

    [SerializeField]
    private SceneInfo[] scenes = default;

    private IEnumerator Start()
    {
        // Need the WebCam Authorization before using Camera on mobile devices
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        }

        // Init scenes
        foreach (var scene in scenes)
        {
            var button = Instantiate(buttonPrefab, parent);
            button.name = scene.displayName;
            button.sceneName = scene.path;
            button.DisplayName = scene.displayName;
        }
    }
}
