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

        public BlazePose(Options options)
        {
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
            poseLandmark.Invoke(texture, poseResult);
            landmarkResult = poseLandmark.GetResult();
            if (landmarkResult.score < 0.3f)
            {
                poseResult.score = landmarkResult.score;
            }
            else
            {
                poseResult = PoseLandmarkDetect.LandmarkToDetection(landmarkResult);
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

            return landmarkResult;
        }
    }
}
