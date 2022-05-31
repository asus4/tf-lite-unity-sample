using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

namespace TensorFlowLite
{
    [RequireComponent(typeof(Button))]
    public class GoToSceneButton : MonoBehaviour
    {
        [System.Serializable]
        public class DisplayNameEvent : UnityEvent<string> { }

        [ScenePath]
        public string sceneName = string.Empty;

        [SerializeField]
        private string displayName = string.Empty;

        public LoadSceneMode mode = LoadSceneMode.Single;

        public DisplayNameEvent onDisplayNameChanged = new DisplayNameEvent();

        public string DisplayName
        {
            get => displayName;
            set
            {
                displayName = value;
                if (!string.IsNullOrWhiteSpace(value))
                {
                    onDisplayNameChanged.Invoke(displayName);
                }
            }
        }

        void OnEnable()
        {
            GetComponent<Button>().onClick.AddListener(OnButtonClick);
            DisplayName = string.IsNullOrWhiteSpace(displayName) ? sceneName : displayName;
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
