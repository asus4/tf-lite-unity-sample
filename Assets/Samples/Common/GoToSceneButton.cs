using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

namespace TensorFlowLite
{
    [RequireComponent(typeof(Button))]
    public class GoToSceneButton : MonoBehaviour
    {
        public string sceneName = "";
        public LoadSceneMode mode = LoadSceneMode.Single;

        void OnEnable()
        {
            GetComponent<Button>().onClick.AddListener(OnButtonClick);
        }

        void OnDisable()
        {
            GetComponent<Button>().onClick.RemoveListener(OnButtonClick);
        }

        void OnButtonClick()
        {
            SceneManager.LoadScene(sceneName, mode);
        }
    }
}
