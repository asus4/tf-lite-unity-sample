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
        }

        // classificators / scores
        private float[] output0 = new float[2944];

        // regressors / points
        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 17 are 7 hand keypoint x and y coordinates: x1,y1,x2,y2,...x7,y7
        private float[,] output1 = new float[2944, 18];
        private List<Palm> results = new List<Palm>();
        private Anchor[] anchors;

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

        public List<Palm> GetResults(float scoreThreshold)
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

                results.Add(new Palm()
                {
                    score = score,
                    rect = new Rect(cx - w * 0.5f, cy - h * 0.5f, w, h),
                });

            }

            return results;
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

    }
}
