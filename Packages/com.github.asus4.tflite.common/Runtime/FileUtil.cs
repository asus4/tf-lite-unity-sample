using System.IO;
using System.Threading;
using UnityEngine;
using UnityEngine.Networking;
#if TFLITE_UNITASK_ENABLED
using Cysharp.Threading.Tasks;
#endif

namespace TensorFlowLite
{
    public static class FileUtil
    {
        public static byte[] LoadFile(string path)
        {
            string uri = GetStreamingAssetsUri(path);
            using (var request = UnityWebRequest.Get(uri))
            {
                request.SendWebRequest();
                while (!request.isDone)
                {
                }
                return request.downloadHandler.data;
            }
        }

#if TFLITE_UNITASK_ENABLED
        public static async UniTask<byte[]> LoadFileAsync(string path, CancellationToken cancellationToken = default)
        {
            string uri = GetStreamingAssetsUri(path);
            using (var request = UnityWebRequest.Get(uri))
            {
                await request.SendWebRequest().WithCancellation(cancellationToken);
                return request.downloadHandler.data;
            }
        }
#endif

        static string GetStreamingAssetsUri(string path)
        {
            if (!IsPathRooted(path))
            {
                path = Path.Combine(Application.streamingAssetsPath, path);
            }

            if (Application.platform != RuntimePlatform.Android)
            {
                path = "file://" + path;
            }
            return path;
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
