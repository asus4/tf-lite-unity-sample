using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace PolyPerfect
{
    public class Common_AnimSpeed : MonoBehaviour
    {

        Animator anim;

        float Speed;

        // Use this for initialization
        void Start()
        {
            Speed = Random.Range(0.85f, 1.25f);
            anim = GetComponent<Animator>();
        }

        // Update is called once per frame
        void Update()
        {
            anim.speed = Speed;
        }
    }
}