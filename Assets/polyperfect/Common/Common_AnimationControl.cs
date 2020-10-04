using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace PolyPerfect
{
    public class Common_AnimationControl : MonoBehaviour
    {
        string currentAnimation = "";
        // Use this for initialization
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {


        }
        public void SetAnimation(string animationName)
        {

            if (currentAnimation != "")
            {
                this.GetComponent<Animator>().SetBool(currentAnimation, false);
            }
            this.GetComponent<Animator>().SetBool(animationName, true);
            currentAnimation = animationName;
        }

        public void SetAnimationIdle()
        {

            if (currentAnimation != "")
            {
                this.GetComponent<Animator>().SetBool(currentAnimation, false);
            }


        }
        public void SetDeathAnimation(int numOfClips)
        {

            int clipIndex = Random.Range(0, numOfClips);
            string animationName = "Death";
            Debug.Log(clipIndex);

            this.GetComponent<Animator>().SetInteger(animationName, clipIndex);
        }
    }
}