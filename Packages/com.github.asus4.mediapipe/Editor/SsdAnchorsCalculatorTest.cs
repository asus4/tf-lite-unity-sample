using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace TensorFlowLite
{
    public class SsdAnchorsCalculatorTest
    {
        [Test]
        public void FaceDetectionConfigTest()
        {
            var options = new SsdAnchorsCalculator.Options()
            {
                numLayers = 5,

                minScale = 0.1171875f,
                maxScale = 0.75f,

                inputSizeHeight = 256,
                inputSizeWidth = 256,

                anchorOffsetX = 0.5f,
                anchorOffsetY = 0.5f,

                featureMapWidth = new int[0] { },
                featureMapHeight = new int[0] { },
                strides = new int[] { 8, 16, 32, 32, 32 },
                aspectRatios = new float[] { 1.0f },

                reduceBoxesInLowestLayer = false,
                interpolatedScaleAspectRatio = 1.0f,
                fixedAnchorSize = true,
            };

            var testData = LoadTestData("anchor_golden_file_0.txt");
            var anchors = SsdAnchorsCalculator.Generate(options);
            AreEqual(testData, anchors);
        }

        [Test]
        public void MobileSSDConfig()
        {
            var options = new SsdAnchorsCalculator.Options()
            {
                numLayers = 6,

                minScale = 0.2f,
                maxScale = 0.95f,

                inputSizeHeight = 300,
                inputSizeWidth = 300,

                anchorOffsetX = 0.5f,
                anchorOffsetY = 0.5f,

                featureMapWidth = new int[0] { },
                featureMapHeight = new int[0] { },
                strides = new int[] { 16, 32, 64, 128, 256, 512 },
                aspectRatios = new float[] { 1.0f, 2.0f, 0.5f, 3.0f, 0.3333f },

                reduceBoxesInLowestLayer = true,
                interpolatedScaleAspectRatio = 1.0f,
                fixedAnchorSize = false,
            };

            var testData = LoadTestData("anchor_golden_file_1.txt");
            var anchors = SsdAnchorsCalculator.Generate(options);
            AreEqual(testData, anchors);
        }

        private static void AreEqual(SsdAnchor[] expected, SsdAnchor[] actual)
        {
            Assert.AreEqual(expected.Length, actual.Length);

            int length = expected.Length;
            for (int i = 0; i < length; i++)
            {
                Assert.IsTrue(expected[i] == actual[i]);
                // Assert.AreEqual(expected[i], actual[i]);
            }
            Debug.Log("all equal");
        }

        private static SsdAnchor[] LoadTestData(string testFile)
        {
            const string testDataPath = "Packages/com.github.asus4.mediapipe/Editor/TestData";
            string path = Path.GetFullPath(Path.Combine(testDataPath, testFile));
            Assert.IsTrue(File.Exists(path));

            var anchors = new List<SsdAnchor>();

            using (var reader = new StreamReader(path))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;

                    string[] chunks = line.Split(' ');
                    anchors.Add(new SsdAnchor()
                    {
                        x = float.Parse(chunks[0]),
                        y = float.Parse(chunks[1]),
                        width = float.Parse(chunks[2]),
                        height = float.Parse(chunks[3]),
                    });
                }
            }
            return anchors.ToArray();
        }
    }
}

