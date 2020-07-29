using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using NUnit.Framework;

namespace TensorFlowLite
{
    public class BertTest
    {
        [Test]
        public void InvokeTest()
        {
            string path = System.IO.Path.Combine(Application.streamingAssetsPath, "mobilebert_float.tflite");
            var vocabText = AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/Samples/Bert/vocab.txt");
            var bert = new Bert(path, vocabText.text);

            string content = "TensorFlow is a free and open-source software library for dataflow and "
                + "differentiable programming across a range of tasks. It is a symbolic math library, and "
                + "is also used for machine learning applications such as neural networks. It is used for "
                + "both research and production at Google. TensorFlow was developed by the Google Brain "
                + "team for internal Google use. It was released under the Apache License 2.0 on November "
                + "9, 2015.";
            
            string question1 = "What is TensorFlow";
            string answer1 = "a free and open-source software library for dataflow and differentiable "
                + "programming across a range of tasks";
            bert.Invoke(question1, content);
            
        }
    }
}
