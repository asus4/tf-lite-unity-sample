using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using NUnit.Framework;

namespace TensorFlowLite
{
    public class BertTest
    {
        private Bert bert;

        [SetUp]
        public void SetUp()
        {
            if (bert != null) return;
            string path = System.IO.Path.Combine(Application.streamingAssetsPath, "mobilebert_float.tflite");
            var vocabText = AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/Samples/Bert/vocab.txt");
            bert = new Bert(path, vocabText.text);
        }

        [TestCase( // 1
            "What is TensorFlow",
            "a free and open-source software library for dataflow and differentiable "
            + "programming across a range of tasks"
        )]
        [TestCase( // 2
            "Who developed TensorFlow?",
            "Google Brain team"
        )]
        [TestCase( // 3
            "When was TensorFlow released?",
            "November 9, 2015"
        )]
        [TestCase( // 4
            "What is TensorFlow used for?",
            "symbolic math library, and is also used for machine learning applications such as neural "
            + "networks"
        )]
        [TestCase( // 5
            "How is TensorFlow used in Google?",
            "both research and production"
        )]
        [TestCase( // 6
            "Which license does TensorFlow use?",
            "Apache License 2.0"
        )]
        public void InvokeTest(string question, string answer)
        {
            const string content = "TensorFlow is a free and open-source software library for dataflow and "
                + "differentiable programming across a range of tasks. It is a symbolic math library, and "
                + "is also used for machine learning applications such as neural networks. It is used for "
                + "both research and production at Google. TensorFlow was developed by the Google Brain "
                + "team for internal Google use. It was released under the Apache License 2.0 on November "
                + "9, 2015.";

            var result = bert.Invoke(question, content);
            ContainsText(answer, result);
        }

        [TestCase( // 1
            "What is Tesla's home country?",
            "Serbian"
        )]
        [TestCase( // 2
            "What was Nikola Tesla's ethnicity?",
            "Serbian"
        )]
        [TestCase( // 3
            "What does AC stand for?",
            "alternating current"
        )]
        [TestCase( // 4
            "When was Tesla born?",
            "10 July 1856"
        )]
        [TestCase( // 5
            "In what year did Tesla die?",
            "1943"
        )]
        public void InvokeUnicodeTest(string question, string answer)
        {
            const string content = "Nikola Tesla (Serbian Cyrillic: \u041d\u0438\u043a\u043e\u043b"
                + "\u0430 \u0422\u0435\u0441\u043b\u0430; 10 July 1856 \u2013 7 January 1943) "
                + "was a Serbian American inventor, electrical engineer, mechanical engineer, physicist, and "
                + "futurist best known for his contributions to the design of the modern alternating current "
                + "(AC) electricity supply system.";
            
             var result = bert.Invoke(question, content);
            ContainsText(answer, result);
        }

        private static void ContainsText(string expected, Bert.Answer[] answers)
        {
            Assert.IsNotNull(answers);
            Assert.True(answers.Length > 0);
            string actual = answers[0].text;
            Assert.True(actual.Contains(expected), $"expected: {expected}\nbut actual: {actual}");
        }
    }
}
