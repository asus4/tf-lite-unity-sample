using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace TensorFlowLite
{

    public class PalmDetect : BaseImagePredictor<float>
    {
        private struct Anchor
        {
            public float x; // center
            public float y; // center
            public float width;
            public float height;
        }

        public struct Palm
        {
            public float score;
            public Rect rect;
            public Vector2[] keypoints;
        }

        public const int MAX_PALM_NUM = 4;

        // classificators / scores
        private float[] output0 = new float[2944];

        // regressors / points
        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 17 are 7 hand keypoint x and y coordinates: x1,y1,x2,y2,...x7,y7
        private float[,] output1 = new float[2944, 18];
        private List<Palm> results = new List<Palm>();
        private Anchor[] anchors;

        public float[,,] Input0 => inputs;

        public PalmDetect(string modelPath, string anchorCSV) : base(modelPath, true)
        {
            // TODO calc anchor with Anchor calclator
            anchors = ParseAnchors(anchorCSV);
            UnityEngine.Debug.AssertFormat(anchors.Length == 2944, "Anchors count must be 2944");
        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, inputs);

            interpreter.SetInputTensorData(0, inputs);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }

        public List<Palm> GetResults(float scoreThreshold = 0.7f, float iouThreshold = 0.3f)
        {
            results.Clear();

            for (int i = 0; i < anchors.Length; i++)
            {
                float score = Sigmoid(output0[i]);
                if (score < scoreThreshold)
                {
                    continue;
                }

                Anchor anchor = anchors[i];

                float sx = output1[i, 0];
                float sy = output1[i, 1];
                float w = output1[i, 2];
                float h = output1[i, 3];

                float cx = sx + anchor.x * width;
                float cy = sy + anchor.y * height;

                cx /= (float)width;
                cy /= (float)height;
                w /= (float)width;
                h /= (float)height;

                var keypoints = new Vector2[7];
                for (int j = 0; j < 7; j++)
                {
                    float lx = output1[i, 4 + (2 * j) + 0];
                    float ly = output1[i, 4 + (2 * j) + 1];
                    lx += anchor.x * width;
                    ly += anchor.y * height;
                    lx /= (float)width;
                    ly /= (float)height;
                    keypoints[j] = new Vector2(lx, ly);
                }

                results.Add(new Palm()
                {
                    score = score,
                    rect = new Rect(cx - w * 0.5f, cy - h * 0.5f, w, h),
                    keypoints = keypoints,
                });

            }

            return NonMaxSuppression(results, iouThreshold);
        }

        private static float Sigmoid(float x)
        {
            return (1.0f / (1.0f + Mathf.Exp(-x)));
        }

        private static Anchor[] ParseAnchors(string csv)
        {
            string[] lines = csv.Split(new char[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
            var anchors = new Anchor[lines.Length];

            var lineSeparator = new char[] { ',' };
            for (int i = 0; i < lines.Length; i++)
            {
                var cols = lines[i].Split(lineSeparator, StringSplitOptions.RemoveEmptyEntries);
                anchors[i] = new Anchor()
                {
                    x = float.Parse(cols[0]),
                    y = float.Parse(cols[1]),
                    width = float.Parse(cols[2]),
                    height = float.Parse(cols[3]),
                };
            }
            return anchors;
        }

        private static List<Palm> NonMaxSuppression(List<Palm> palms, float iou_threshold)
        {
            var filtered = new List<Palm>();

            foreach (Palm originalPalm in palms.OrderByDescending(o => o.score))
            {
                bool ignore_candidate = false;
                foreach (Palm newPalm in filtered)
                {
                    float iou = CalcIntersectionOverUnion(originalPalm.rect, newPalm.rect);
                    if (iou >= iou_threshold)
                    {
                        ignore_candidate = true;
                        break;
                    }
                }

                if (!ignore_candidate)
                {
                    filtered.Add(originalPalm);
                    if (filtered.Count >= MAX_PALM_NUM)
                    {
                        break;
                    }
                }
            }

            return filtered;
        }

        private static float CalcIntersectionOverUnion(Rect rect0, Rect rect1)
        {
            float sx0 = rect0.xMin;
            float sy0 = rect0.yMin;
            float ex0 = rect0.xMax;
            float ey0 = rect0.yMax;
            float sx1 = rect1.xMin;
            float sy1 = rect1.yMin;
            float ex1 = rect1.xMax;
            float ey1 = rect1.yMax;

            float xmin0 = Mathf.Min(sx0, ex0);
            float ymin0 = Mathf.Min(sy0, ey0);
            float xmax0 = Mathf.Max(sx0, ex0);
            float ymax0 = Mathf.Max(sy0, ey0);
            float xmin1 = Mathf.Min(sx1, ex1);
            float ymin1 = Mathf.Min(sy1, ey1);
            float xmax1 = Mathf.Max(sx1, ex1);
            float ymax1 = Mathf.Max(sy1, ey1);

            float area0 = (ymax0 - ymin0) * (xmax0 - xmin0);
            float area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
            if (area0 <= 0 || area1 <= 0)
            {
                return 0.0f;
            }

            float intersect_xmin = Mathf.Max(xmin0, xmin1);
            float intersect_ymin = Mathf.Max(ymin0, ymin1);
            float intersect_xmax = Mathf.Min(xmax0, xmax1);
            float intersect_ymax = Mathf.Min(ymax0, ymax1);

            float intersect_area = Mathf.Max(intersect_ymax - intersect_ymin, 0.0f) *
                                   Mathf.Max(intersect_xmax - intersect_xmin, 0.0f);

            return intersect_area / (area0 + area1 - intersect_area);
        }


    }
}
