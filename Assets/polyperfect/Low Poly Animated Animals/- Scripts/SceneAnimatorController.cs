using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SceneAnimatorController : MonoBehaviour
{
    Animator animator;

    void Start()
    {
        animator = GetComponent<Animator>();
    }

    public void SetAnimatorString(string value)
    {
        for (int i = 0; i < animator.parameterCount; i++)
        {
            animator.SetBool(animator.parameters[i].name, false);
        }

        if(value != "Idle")
        animator.SetBool(value, true);
    }
}
