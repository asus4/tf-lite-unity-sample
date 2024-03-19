using System;
using System.Buffers;
using System.Collections;
using System.Linq;
using UnityEngine;
using Object = UnityEngine.Object;

namespace TensorFlowLite
{
    /// <summary>
    /// An utility for getting latest N samples from microphone.
    /// </summary>
    public class MicrophoneBuffer : IDisposable
    {
        public enum Frequency
        {
            Hz8000 = 8000,
            Hz16000 = 16000,
            Hz32000 = 32000,
            Hz44100 = 44100,
            Hz48000 = 48000,
            Hz96000 = 96000,
        }

        [Serializable]
        public class Options
        {
            [Tooltip("Microphone device name; null to use default device")]
            public string deviceName = null;

            [Tooltip("Mic Frequency in Hz")]
            public Frequency frequency = Frequency.Hz16000;

            [Tooltip("Max duration in seconds")]
            [Min(1)]
            public int maxDurationSec = 5;
        }

        [Tooltip("Default options")]
        [SerializeField]
        private Options defaultOptions = new();

        private string deviceName;
        private AudioClip clip;
        private ArrayPool<float> pool;

        public bool IsRecording => clip != null;

        public void Dispose()
        {
            StopRecording();
        }

        public IEnumerator StartRecording(Options options = null)
        {
            // Check permission
            yield return Application.RequestUserAuthorization(UserAuthorization.Microphone);
            if (!Application.HasUserAuthorization(UserAuthorization.Microphone))
            {
                Debug.LogError("Microphone permission is required");
                yield break;
            }

            options ??= defaultOptions;

            // Find device name
            string[] availableDevices = Microphone.devices;
            deviceName = options.deviceName;
            if (string.IsNullOrEmpty(deviceName) || !availableDevices.Contains(deviceName))
            {
                deviceName = availableDevices[0];
            }

            // Limit frequency
            int frequency = (int)options.frequency;
            Microphone.GetDeviceCaps(deviceName, out int minFreq, out int maxFreq);
            frequency = Math.Clamp(frequency, minFreq, maxFreq);

            int maxDurationSec = Math.Max(1, options.maxDurationSec);

            var clip = Microphone.Start(deviceName, true, maxDurationSec, frequency);
            pool ??= ArrayPool<float>.Create(frequency * maxDurationSec, 2);

            // Wait until recording starts
            yield return new WaitUntil(() => Microphone.GetPosition(deviceName) > 0);
            this.clip = clip;

            Debug.Log($"Started Microphone: Device={deviceName}, Hz={frequency}, MaxSec={maxDurationSec}");
        }

        public void StopRecording()
        {
            {
                Microphone.End(deviceName);
            }
            if (clip != null)
            {
                Object.Destroy(clip);
                clip = null;
            }
        }

        public void GetLatestSamples(float[] samples)
        {
            if (!IsRecording)
            {
                throw new InvalidOperationException("Recording is not started");
            }
            if (samples.Length > clip.samples)
            {
                throw new ArgumentException("samples.Length must be less than clip total samples");
            }

            int position = Microphone.GetPosition(deviceName);
            if (position < samples.Length)
            {
                // sample from first and last then concat
                Debug.Log("TODO");
            }
            else
            {
                int offset = position - samples.Length;
                clip.GetData(samples, offset);
            }
        }
    }
}
