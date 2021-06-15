using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace TensorFlowLite
{
    public static class FileUtil
    {
        public static byte[] LoadFile(string path)
        {
            if (!IsPathRooted(path))
            {
                path = Path.Combine(Application.streamingAssetsPath, path);
            }

            if (Application.platform != RuntimePlatform.Android)
            {
                path = "file://" + path;
            }
            using (var request = UnityWebRequest.Get(path))
            {
                request.SendWebRequest();
                while (!request.isDone)
                {
                }
                return request.downloadHandler.data;
            }
        }

        static bool IsPathRooted(string path)
        {
            if (path.StartsWith("jar:file:"))
            {
                return true;
            }
            return Path.IsPathRooted(path);
        }
    }
}
