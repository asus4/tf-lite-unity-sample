using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;

/* ---------------------------------------------------------------------- *
 *   mediapipe/calculators/tflite/ssd_anchors_calculator_test.cc
 * ---------------------------------------------------------------------- */

// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc

namespace TensorFlowLite
{
    public struct SsdAnchor : System.IEquatable<SsdAnchor>
    {
        public float x; // center
        public float y; // center
        public float width;
        public float height;



        public override int GetHashCode()
        {
            return x.GetHashCode() ^ (y.GetHashCode() << 2) ^ (width.GetHashCode() >> 2) ^ (height.GetHashCode() >> 1);
        }

        public override bool Equals(object other)
        {
            if (!(other is SsdAnchor)) return false;
            return Equals((SsdAnchor)other);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(SsdAnchor other)
        {
            return x == other.x
                && y == other.y
                && width == other.width
                && height == other.height;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator ==(SsdAnchor lhs, SsdAnchor rhs)
        {
            // Returns false in the presence of NaN values.
            float diffx = lhs.x - rhs.x;
            float diffy = lhs.y - rhs.y;
            float diffw = lhs.width - rhs.width;
            float diffh = lhs.height - rhs.height;
            float sqrmag = diffx * diffx + diffy * diffy + diffw * diffw + diffw * diffw;
            return sqrmag < kEpsilon * kEpsilon;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator !=(SsdAnchor lhs, SsdAnchor rhs)
        {
            // Returns true in the presence of NaN values.
            return !(lhs == rhs);
        }

        public const float kEpsilon = 0.00001F;


        public override string ToString()
        {
            return $"({x},{y},{width},{height})";
        }

    }

    public static class SsdAnchorsCalculator
    {

        // SsdAnchorsCalculatorOptions
        public struct Options
        {
            public int inputSizeWidth;
            public int inputSizeHeight;

            public float minScale;
            public float maxScale;

            public float anchorOffsetX;
            public float anchorOffsetY;

            public int numLayers;
            public int[] featureMapWidth;
            public int[] featureMapHeight;
            public int[] strides;

            public float[] aspectRatios;

            public bool reduceBoxesInLowestLayer;
            public float interpolatedScaleAspectRatio;

            public bool fixedAnchorSize;
        }

        private static float CalculateScale(float minScale, float maxScale, int strideIndex, int numStrides)
        {
            return minScale + (maxScale - minScale) * 1.0f * strideIndex / (numStrides - 1.0f);
        }

        public static SsdAnchor[] Generate(Options options)
        {
            var anchors = new List<SsdAnchor>();

            int layer_id = 0;
            while (layer_id < options.strides.Length)
            {
                var anchor_height = new List<float>();
                var anchor_width = new List<float>();
                var aspect_ratios = new List<float>();
                var scales = new List<float>();

                // For same strides, we merge the anchors in the same order.
                int last_same_stride_layer = layer_id;
                while (last_same_stride_layer < (int)options.strides.Length
                       && options.strides[last_same_stride_layer] == options.strides[layer_id])
                {
                    float scale = CalculateScale(options.minScale,
                                                 options.maxScale,
                                                 last_same_stride_layer,
                                                 options.strides.Length);
                    if (last_same_stride_layer == 0 && options.reduceBoxesInLowestLayer)
                    {
                        // For first layer, it can be specified to use predefined anchors.
                        aspect_ratios.Add(1.0f);
                        aspect_ratios.Add(2.0f);
                        aspect_ratios.Add(0.5f);
                        scales.Add(0.1f);
                        scales.Add(scale);
                        scales.Add(scale);
                    }
                    else
                    {
                        for (int aspect_ratio_id = 0;
                            aspect_ratio_id < (int)options.aspectRatios.Length;
                             ++aspect_ratio_id)
                        {
                            aspect_ratios.Add(options.aspectRatios[aspect_ratio_id]);
                            scales.Add(scale);
                        }
                        if (options.interpolatedScaleAspectRatio > 0.0)
                        {
                            float scale_next = last_same_stride_layer == (int)options.strides.Length - 1
                                    ? 1.0f
                                    : CalculateScale(options.minScale, options.maxScale,
                                                     last_same_stride_layer + 1,
                                                     options.strides.Length);
                            scales.Add(Mathf.Sqrt(scale * scale_next));
                            aspect_ratios.Add(options.interpolatedScaleAspectRatio);
                        }
                    }
                    last_same_stride_layer++;
                }

                for (int i = 0; i < (int)aspect_ratios.Count; ++i)
                {
                    float ratio_sqrts = Mathf.Sqrt(aspect_ratios[i]);
                    anchor_height.Add(scales[i] / ratio_sqrts);
                    anchor_width.Add(scales[i] * ratio_sqrts);
                }

                int feature_map_height = 0;
                int feature_map_width = 0;
                if (options.featureMapHeight.Length > 0)
                {
                    feature_map_height = options.featureMapHeight[layer_id];
                    feature_map_width = options.featureMapWidth[layer_id];
                }
                else
                {
                    int stride = options.strides[layer_id];
                    feature_map_height = Mathf.CeilToInt(1.0f * options.inputSizeHeight / stride);
                    feature_map_width = Mathf.CeilToInt(1.0f * options.inputSizeWidth / stride);
                }

                for (int y = 0; y < feature_map_height; ++y)
                {
                    for (int x = 0; x < feature_map_width; ++x)
                    {
                        for (int anchor_id = 0; anchor_id < (int)anchor_height.Count; ++anchor_id)
                        {
                            // TODO: Support specifying anchor_offset_x, anchor_offset_y.
                            float x_center = (x + options.anchorOffsetX) * 1.0f / feature_map_width;
                            float y_center = (y + options.anchorOffsetY) * 1.0f / feature_map_height;

                            var new_anchor = new SsdAnchor();
                            new_anchor.x = x_center;
                            new_anchor.y = y_center;

                            if (options.fixedAnchorSize)
                            {
                                new_anchor.width = 1.0f;
                                new_anchor.height = 1.0f;
                            }
                            else
                            {
                                new_anchor.width = anchor_width[anchor_id];
                                new_anchor.height = anchor_height[anchor_id];
                            }
                            anchors.Add(new_anchor);
                        }
                    }
                }
                layer_id = last_same_stride_layer;
            }
            return anchors.ToArray();
        }
    }
}

