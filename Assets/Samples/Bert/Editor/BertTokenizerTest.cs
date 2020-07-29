using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEditor;
using NUnit.Framework;

namespace TensorFlowLite
{
    public class BertTokenizerTest
    {
        private Dictionary<string, int> vocabularyTable;

        [SetUp]
        public void SetUp()
        {
            var vocabText = AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/Samples/Bert/vocab.txt");
            Debug.Log($"vocab: {vocabText}");
            vocabularyTable = Bert.LoadVocabularies(vocabText.text);
        }

        [Test]
        public void WhiteSpaceTest()
        {
            Assert.True(' '.IsBertWhiteSpace());
            Assert.True('\t'.IsBertWhiteSpace());
            Assert.True('\r'.IsBertWhiteSpace());
            Assert.True('\n'.IsBertWhiteSpace());
            Assert.True('\u00A0'.IsBertWhiteSpace());

            Assert.False('A'.IsBertWhiteSpace());
            Assert.False('-'.IsBertWhiteSpace());
        }

        [Test]
        public void ControlTest()
        {
            Assert.True('\u0005'.IsBertControl());

            Assert.False('A'.IsBertControl());
            Assert.False(' '.IsBertControl());
            Assert.False('\t'.IsBertControl());
            Assert.False('\r'.IsBertControl());
            Assert.False("\u1F4A9".IsBertControl());
        }

        [Test]
        public void PunctuationTest()
        {
            Assert.True('-'.IsBertPunctuation());
            Assert.True('$'.IsBertPunctuation());
            Assert.True('`'.IsBertPunctuation());
            Assert.True('.'.IsBertPunctuation());

            Assert.False('A'.IsBertPunctuation());
            Assert.False(' '.IsBertPunctuation());
        }

        [TestCase("  Hi, This\tis an example.\n",
        new string[] { "hi", ",", "this", "is", "an", "example", "." })]
        [TestCase("Hello,How are you?",
        new string[] { "hello", ",", "how", "are", "you", "?" })]
        public void BasicTokenizeTest(string input, string[] expected)
        {
            string[] result = BertTokenizer.BasicTokenize(input);
            ArrayEqual(expected, result);
        }


        [TestCase("Good morning, I'm your teacher.\n",
        new string[] { "good", "morning", ",", "i", "'", "m", "your", "teacher", "." })]
        [TestCase("Nikola Tesla\t(Serbian Cyrillic: 10 July 1856 ~ 7 January 1943)",
        new string[] { "nikola", "tesla", "(", "serbian", "cyrillic", ":", "10", "july", "1856", "~", "7", "january", "1943", ")" })]
        public void FullTokenizeTest(string input, string[] expected)
        {
            string[] result = BertTokenizer.BasicTokenize(input);
            ArrayEqual(expected, result);
        }

        private static void ArrayEqual(string[] a, string[] b)
        {
            Assert.AreEqual(a.Length, b.Length);
            for (int i = 0; i < a.Length; i++)
            {
                Assert.AreEqual(a[i], b[i]);
            }
        }

    }
}
