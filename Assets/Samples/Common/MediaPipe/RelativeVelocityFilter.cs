using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// RelativeVelocityFilter from MediaPipe
    /// mediapipe/graphs/pose_tracking/calculators/relative_velocity_filter.h
    /// 
    /// This filter keeps track (on a window of specified size) of
    /// value changes over time, which as result gives us velocity of how value
    /// changes over time. With higher velocity it weights new values higher.
    ///
    /// Use @window_size and @velocity_scale to tweak this filter for your use case.
    ///
    /// - higher @window_size adds to lag and to stability
    /// - lower @velocity_scale adds to lag and to stability
    /// </summary>
    public class RelativeVelocityFilter
    {
        public enum DistanceEstimationMode
        {
            // When the value scale changes, uses a heuristic
            // that is not translation invariant (see the implementation for details).
            LegacyTransition,
            // The current (i.e. last) value scale is always used for scale estimation.
            // When using this mode, the filter is translation invariant, i.e.
            //     Filter(Data + Offset) = Filter(Data) + Offset.
            ForceCurrentScale,
        }

        private struct WindowElement
        {
            public float distance;
            public double duration;
            public WindowElement(float distance, double duration)
            {
                this.distance = distance;
                this.duration = duration;
            }
        };

        private float lastValue = 0.0f;
        private float lastValueScale = 1.0f;
        private double lastTimestamp = -1;

        private uint maxWindowSize;
        private Queue<WindowElement> windows;
        private LowPassFilter lowPassFilter;
        private float velocityScale;
        private DistanceEstimationMode distanceMode;

        public float VelocitySacle
        {
            get => velocityScale;
            set => velocityScale = value;
        }

        public RelativeVelocityFilter(
            uint windowSize,
            float velocityScale,
            DistanceEstimationMode distanceMode)
        {
            maxWindowSize = windowSize;
            this.velocityScale = velocityScale;
            this.distanceMode = distanceMode;
            lowPassFilter = new LowPassFilter()
            {
                alpha = 1f,
            };
            windows = new Queue<WindowElement>();
        }

        public float Apply(double newTimestamp, float valueScale, float value)
        {
            if (lastTimestamp >= newTimestamp)
            {
                // Results are unpredictable in this case, so nothing to do but
                // return same value
                Debug.LogWarning("New timestamp is equal or less than the last one.");
                return value;
            }

            double alpha;
            if (lastTimestamp == -1)
            {
                alpha = 1.0;
            }
            else
            {
                float distance = distanceMode == DistanceEstimationMode.LegacyTransition
                    ? value * valueScale - lastValue * lastValueScale // Original.
                    : valueScale * (value - lastValue);  // Translation invariant.

                double duration = newTimestamp - lastTimestamp;

                float cumulative_distance = distance;
                double cumulative_duration = duration;

                // Define max cumulative duration assuming
                // 30 frames per second is a good frame rate, so assuming 30 values
                // per second or 1 / 30 of a second is a good duration per window element
                const double kAssumedMaxDuration = 1.0 / 30.0;
                double max_cumulative_duration = (1 + windows.Count) * kAssumedMaxDuration;
                foreach (var windows in windows)
                {
                    if (cumulative_duration + windows.duration > max_cumulative_duration)
                    {
                        // This helps in cases when durations are large and outdated
                        // window elements have bad impact on filtering results
                        break;
                    }
                    cumulative_distance += windows.distance;
                    cumulative_duration += windows.duration;
                }
                double velocity = cumulative_distance / cumulative_duration;
                alpha = 1.0 - 1.0 / (1.0 + velocityScale * Math.Abs(velocity));

                windows.Enqueue(new WindowElement(distance, duration));
                if (windows.Count > maxWindowSize)
                {
                    windows.Dequeue();
                }
            }

            lastValue = value;
            lastValueScale = valueScale;
            lastTimestamp = newTimestamp;

            // Debug.Log($"alpha: {alpha}");
            return lowPassFilter.Apply(value, (float)alpha);
        }
    }

    public class RelativeVelocityFilter2D
    {
        private RelativeVelocityFilter x;
        private RelativeVelocityFilter y;

        public float VelocityScale
        {
            get => x.VelocitySacle;
            set
            {
                x.VelocitySacle = value;
                y.VelocitySacle = value;
            }
        }

        public RelativeVelocityFilter2D(
            uint windowSize,
            float velocityScale,
            RelativeVelocityFilter.DistanceEstimationMode distanceMode)
        {
            x = new RelativeVelocityFilter(windowSize, velocityScale, distanceMode);
            y = new RelativeVelocityFilter(windowSize, velocityScale, distanceMode);
        }

        public Vector2 Apply(double newTimestamp, float valueScale, Vector2 value)
        {
            return new Vector2(
                x.Apply(newTimestamp, valueScale, value.x),
                y.Apply(newTimestamp, valueScale, value.y)
            );
        }
    }

    public class RelativeVelocityFilter3D
    {
        private RelativeVelocityFilter x;
        private RelativeVelocityFilter y;
        private RelativeVelocityFilter z;

        public float VelocityScale
        {
            get => x.VelocitySacle;
            set
            {
                x.VelocitySacle = value;
                y.VelocitySacle = value;
                z.VelocitySacle = value;
            }
        }

        public RelativeVelocityFilter3D(
            uint windowSize,
            float velocityScale,
            RelativeVelocityFilter.DistanceEstimationMode distanceMode)
        {
            x = new RelativeVelocityFilter(windowSize, velocityScale, distanceMode);
            y = new RelativeVelocityFilter(windowSize, velocityScale, distanceMode);
            z = new RelativeVelocityFilter(windowSize, velocityScale, distanceMode);
        }

        public Vector3 Apply(double newTimestamp, float valueScale, Vector3 value)
        {
            return new Vector3(
                x.Apply(newTimestamp, valueScale, value.x),
                y.Apply(newTimestamp, valueScale, value.y),
                z.Apply(newTimestamp, valueScale, value.z)
            );
        }
    }
}
