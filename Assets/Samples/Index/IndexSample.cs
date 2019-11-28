using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class IndexSample : MonoBehaviour
{
    IEnumerator Start()
    {
        // Need the WebCam Authorization before using Camera on mobile devices
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        }
    }

}
