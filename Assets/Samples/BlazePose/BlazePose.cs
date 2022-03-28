using System;
using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;

namespace TensorFlowLite
{
    public class BlazePose : IDisposable
    {
        [Serializable]
        public class Options
        {
            public PoseDetect.Options detect;
            public PoseLandmarkDetect.Options landmark;
        }

        private readonly PoseDetect poseDetect;
        private readonly PoseLandmarkDetect poseLandmark;

        private PoseDetect.Result poseResult = default;
        private PoseLandmarkDetect.Result landmarkResult = default;

        private bool NeedsDetectionUpdate => poseResult == null || poseResult.score < 0.5f;

        public Matrix4x4 CropMatrix => poseLandmark.CropMatrix;

        public PoseDetect.Result PoseResult => poseResult;
        public PoseLandmarkDetect.Result LandmarkResult => landmarkResult;
        public Texture LandmarkInputTexture => poseLandmark.inputTex;

        private readonly Options options;

        public BlazePose(Options options)
        {
            this.options = options;
            // Sync aspect mode from detect
            options.landmark.AspectMode = options.detect.aspectMode;

            poseDetect = new PoseDetect(options.detect);
            poseLandmark = new PoseLandmarkDetect(options.landmark);
        }

        public void Dispose()
        {
            poseDetect?.Dispose();
            poseLandmark?.Dispose();
        }

        public PoseLandmarkDetect.Result Invoke(Texture texture)
        {
            if (NeedsDetectionUpdate)
            {
                poseDetect.Invoke(texture);
                poseResult = poseDetect.GetResults();
            }
            if (poseResult.score < 0)
            {
                poseResult = null;
                landmarkResult = null;
                return null;
            }
            landmarkResult = poseLandmark.Invoke(texture, poseResult);
            if (landmarkResult.score < 0.3f)
            {
                poseResult.score = landmarkResult.score;
            }
            else
            {
                poseResult = PoseLandmarkDetect.LandmarkToDetection(landmarkResult);
            }

            if (landmarkResult != null)
            {
                CorrectViewportLandmarks(landmarkResult.viewportLandmarks, options.landmark.AspectMode);
            }
            return landmarkResult;
        }

        public async UniTask<PoseLandmarkDetect.Result> InvokeAsync(Texture texture, CancellationToken cancellationToken)
        {
            if (NeedsDetectionUpdate)
            {
                // Note: `await` changes PlayerLoopTiming from Update to FixedUpdate.
                poseResult = await poseDetect.InvokeAsync(texture, cancellationToken, PlayerLoopTiming.FixedUpdate);
            }
            if (poseResult.score < 0)
            {
                poseResult = null;
                landmarkResult = null;
                return null;
            }

            landmarkResult = await poseLandmark.InvokeAsync(texture, poseResult, cancellationToken, PlayerLoopTiming.Update);

            // Generate poseResult from landmarkResult
            if (landmarkResult.score < 0.3f)
            {
                poseResult.score = landmarkResult.score;
            }
            else
            {
                poseResult = PoseLandmarkDetect.LandmarkToDetection(landmarkResult);
            }

            if (landmarkResult != null)
            {
                CorrectViewportLandmarks(landmarkResult.viewportLandmarks, options.landmark.AspectMode);
            }

            return landmarkResult;
        }


        private static void CorrectViewportLandmarks(Vector4[] landmarks, AspectMode aspectMode)
        {
            if (aspectMode == AspectMode.None)
            {
                // Nothing to do
                return;
            }

            (Vector2 min, Vector2 max) = GetViewportSize(aspectMode);

            // Update world joints
            for (int i = 0; i < landmarks.Length; i++)
            {
                Vector2 p = MathTF.LerpUnclamped(min, max, (Vector2)landmarks[i]);
                // w is visibility
                landmarks[i] = new Vector4(p.x, p.y, landmarks[i].z, landmarks[i].w);
            }
        }

        private static Tuple<Vector2, Vector2> GetViewportSize(AspectMode aspectMode)
        {
            float w = Screen.width;
            float h = Screen.height;
            float aspect = w / h;

            Vector2 min, max;
            switch (aspectMode)
            {
                case AspectMode.Fit:
                    if (aspect > 1)
                    {
                        float n = (w - h) / 2;
                        min = new Vector2(0f, -n);
                        max = new Vector2(w, n + h);
                    }
                    else
                    {
                        float n = (h - w) / 2f;
                        min = new Vector2(-n, 0);
                        max = new Vector2(w + n, h);
                    }
                    break;

                case AspectMode.Fill:
                    if (aspect > 1)
                    {
                        float n = (w - h) / 2;
                        min = new Vector2(n, 0f);
                        max = new Vector2(n + w, h);
                    }
                    else
                    {
                        float n = (h - w) / 2;
                        min = new Vector2(0f, n);
                        max = new Vector2(w, n + w);
                    }
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(aspectMode));
            };

            Vector2 scale = new Vector2(1 / w, 1 / h);
            min = Vector2.Scale(min, scale);
            max = Vector2.Scale(max, scale);

            return new Tuple<Vector2, Vector2>(min, max);
        }
    }
}
