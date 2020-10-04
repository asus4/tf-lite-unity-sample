using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ToggleEvent : MonoBehaviour {

    public UnityEngine.Events.UnityEvent toggleOn, toggleOff;
    UnityEngine.UI.Toggle toggle;

    void Awake()
    {
        toggle = GetComponent<UnityEngine.UI.Toggle>();

        toggle.onValueChanged.AddListener((value) => SwapToggle(toggle.isOn));
    }

    public void SwapToggle(bool value)
    {
        switch (value)
        {
            case true:
                toggleOn.Invoke();
                break;
            case false:
                toggleOff.Invoke();
                break;
        }
    }


}
