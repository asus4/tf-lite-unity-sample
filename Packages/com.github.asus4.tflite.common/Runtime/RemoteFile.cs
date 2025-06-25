// Requires UniTask support
#if TFLITE_UNITASK_ENABLED

using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using UnityEngine;
using UnityEngine.Networking;
using Cysharp.Threading.Tasks;

namespace TensorFlowLite
{
    /// <summary>
    /// Simple remote file download and cache system.
    /// Not for production use.
    /// </summary>
    [Serializable]
    public class RemoteFile : IProgress<float>
    {
        public enum DownloadLocation
        {
            Persistent,
            Cache,
        }

        public string url;
        public DownloadLocation downloadLocation = DownloadLocation.Persistent;

        public event Action<float> OnDownloadProgress;

        public string LocalPath
        {
            get
            {
                string dir = downloadLocation switch
                {
                    DownloadLocation.Persistent => Application.persistentDataPath,
                    DownloadLocation.Cache => Application.temporaryCachePath,
                    _ => throw new Exception($"Unknown download location {downloadLocation}"),
                };
                // make hash from url
                string ext = GetExtension(url);
                string fileName = $"{url.GetHashCode():X8}{ext}";
                return Path.Combine(dir, fileName);
            }
        }

        public bool HasCache => File.Exists(LocalPath);

        public RemoteFile() { }

        public RemoteFile(string url, DownloadLocation location = DownloadLocation.Persistent)
        {
            this.url = url;
            downloadLocation = location;
        }

        // IProgress<float>
        public void Report(float value)
        {
            OnDownloadProgress?.Invoke(value);
        }

        public async UniTask<byte[]> Load(CancellationToken cancellationToken)
        {
            string localPath = LocalPath;

            if (HasCache)
            {
                Log($"Cache Loading file from local: {localPath}");
                using var handler = new DownloadHandlerBuffer();
                if (!localPath.StartsWith("file:/"))
                {
                    localPath = $"file://{localPath}";
                }
                using var request = new UnityWebRequest(localPath, "GET", handler, null);
                this.Report(0.0f);
                await request.SendWebRequest().ToUniTask(progress: this, cancellationToken: cancellationToken);
                this.Report(1.0f);
                return handler.data;
            }
            else
            {
                Log($"Cache not found at {localPath}. Loading from: {url}");
                using var handler = new DownloadHandlerFile(localPath);
                handler.removeFileOnAbort = true;
                using var request = new UnityWebRequest(url, "GET", handler, null);
                this.Report(0.0f);
                await request.SendWebRequest().ToUniTask(progress: this, cancellationToken: cancellationToken);
                this.Report(1.0f);
                return await File.ReadAllBytesAsync(localPath, cancellationToken);
            }
        }

        static string GetExtension(string url)
        {
            string ext = Path.GetExtension(url);
            if (ext.Contains('?'))
            {
                ext = ext[..ext.IndexOf('?')];
            }
            return ext;
        }

        [Conditional("DEVELOPMENT_BUILD"), Conditional("UNITY_EDITOR")]
        static void Log(string message)
        {
            UnityEngine.Debug.Log(message);
        }
    }
}
#endif // TFLITE_UNITASK_ENABLED