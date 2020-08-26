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
            kLegacyTransition,
            // The current (i.e. last) value scale is always used for scale estimation.
            // When using this mode, the filter is translation invariant, i.e.
            //     Filter(Data + Offset) = Filter(Data) + Offset.
            kForceCurrentScale,
        }

        private struct WindowElement
        {
            public float distance;
            public long duration;
            public WindowElement(float distance, long duration)
            {
                this.distance = distance;
                this.duration = duration;
            }
        };

        private float last_value_ = 0.0f;
        private float last_value_scale_ = 1.0f;
        private long last_timestamp_ = -1;

        private uint max_window_size_;
        private Queue<WindowElement> window_;
        private LowPassFilter low_pass_filter_;
        private float velocity_scale_;
        private DistanceEstimationMode distance_mode_;

        public RelativeVelocityFilter(
            uint window_size,
            float velocity_scale,
            DistanceEstimationMode distance_mode)
        {
            max_window_size_ = window_size;
            velocity_scale_ = velocity_scale;
            distance_mode_ = distance_mode;
            low_pass_filter_ = new LowPassFilter()
            {
                alpha = 1f,
            };
            window_ = new Queue<WindowElement>();
        }

        public float Apply(long new_timestamp, float value_scale, float value)
        {
            if (last_timestamp_ >= new_timestamp)
            {
                // Results are unpredictable in this case, so nothing to do but
                // return same value
                Debug.LogWarning("New timestamp is equal or less than the last one.");
                return value;
            }

            float alpha;
            if (last_timestamp_ == -1)
            {
                alpha = 1.0f;
            }
            else
            {
                float distance = distance_mode_ == DistanceEstimationMode.kLegacyTransition
                    ? value * value_scale - last_value_ * last_value_scale_ // Original.
                    : value_scale * (value - last_value_);  // Translation invariant.

                long duration = new_timestamp - last_timestamp_;

                float cumulative_distance = distance;
                long cumulative_duration = duration;

                // Define max cumulative duration assuming
                // 30 frames per second is a good frame rate, so assuming 30 values
                // per second or 1 / 30 of a second is a good duration per window element
                const long kAssumedMaxDuration = 1000000000 / 30;
                long max_cumulative_duration = (1 + window_.Count) * kAssumedMaxDuration;
                foreach (var el in window_)
                {
                    if (cumulative_duration + el.duration > max_cumulative_duration)
                    {
                        // This helps in cases when durations are large and outdated
                        // window elements have bad impact on filtering results
                        break;
                    }
                    cumulative_distance += el.distance;
                    cumulative_duration += el.duration;
                }
                const double kNanoSecondsToSecond = 1e-9;
                float velocity = (float)(cumulative_distance / (cumulative_duration * kNanoSecondsToSecond));
                alpha = 1.0f - 1.0f / (1.0f + velocity_scale_ * Math.Abs(velocity));
                window_.Enqueue(new WindowElement(distance, duration));
                if (window_.Count > max_window_size_)
                {
                    window_.Dequeue();
                }
            }

            last_value_ = value;
            last_value_scale_ = value_scale;
            last_timestamp_ = new_timestamp;

            return low_pass_filter_.Apply(value, alpha);
        }
    }
}
