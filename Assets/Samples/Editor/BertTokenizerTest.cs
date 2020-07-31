using System;
using System.Collections.Generic;
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
            if (vocabularyTable != null)
            {
                return;
            }
            var vocabText = AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/Samples/Bert/vocab.txt");
            vocabularyTable = Bert.LoadVocabularies(vocabText.text);
            // Debug.Log("Vocab Loaded");
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
            string[] result = BertTokenizer.Fulltokenize(input, vocabularyTable);
            ArrayEqual(expected, result);
        }

        [TestCase("", new string[] { })]
        [TestCase("teacher", new string[] { "teacher" })]
        [TestCase("meaningfully", new string[] { "meaningful", "##ly" })]
        public void WordPieceTokenizeTest(string input, string[] expected)
        {
            ArrayEqual(expected, BertTokenizer.WordPieceTokenize(input, vocabularyTable));
        }


        [TestCase("", new string[] { })]
        [TestCase("unwanted running", new string[] { "un", "##want", "##ed", "runn", "##ing" })]
        [TestCase("unwantedX running", new string[] { "[UNK]", "runn", "##ing" })]
        public void WordPieceTokenizeWithCutomCocabTest(string input, string[] expected)
        {
            var vocabText = @"[UNK]
[CLS]
[SEP]
want
##want
##ed
wa
un
runn
##ing";
            var table = Bert.LoadVocabularies(vocabText);

            Assert.True(table.ContainsKey("[UNK]"));
            Assert.True(table.ContainsKey("want"));
            Assert.True(table.ContainsKey("##want"));
            ArrayEqual(expected, BertTokenizer.WordPieceTokenize(input, table));
        }
        internal static void ArrayEqual(string[] a, string[] b)
        {
            Assert.AreEqual(a.Length, b.Length,
                            "a:{0} b:{1}", string.Join("/", a), string.Join("/", b));
            for (int i = 0; i < a.Length; i++)
            {
                Assert.AreEqual(a[i], b[i]);
            }
        }

    }

    public class CharExtensionTest
    {
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
    }

    public class StringExtensionTest
    {
        [Test]
        public void SplitTest()
        {
            BertTokenizerTest.ArrayEqual(
                new string[] { "abcd" },
                "abcd".Split((char c) => false, StringSplitOptions.None)
            );

            BertTokenizerTest.ArrayEqual(
                new string[] { "", "", "", "" },
                "abcd".Split((char c) => true, StringSplitOptions.None)
            );

            BertTokenizerTest.ArrayEqual(
                new string[] { },
                "abcd".Split((char c) => true, StringSplitOptions.RemoveEmptyEntries)
            );

            BertTokenizerTest.ArrayEqual(
                new string[] { "a", "d" },
                "abcd".Split((char c) => c == 'b' || c == 'c', StringSplitOptions.RemoveEmptyEntries)
            );
        }
    }
}
