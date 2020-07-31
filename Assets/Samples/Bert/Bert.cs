using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;


namespace TensorFlowLite
{

    public class Bert : IDisposable
    {

        public class ContentData
        {
            public string[] contentWords;
            public Dictionary<int, int> tokenIdxToWordIdxMapping;
            public string originalContent;
        }

        public struct Score
        {
            public float logit;
            public int start;
            public int end;
        }

        public class Answer
        {
            public string text;
            public Score score;
        }

        const int MAX_ANSWER_LENTH = 32;
        const int MAX_QUERY_LENTH = 64;
        const int MAX_SEQ_LENTH = 384;


        Interpreter interpreter;

        int[] inputs0; // input_ids
        int[] inputs1; // input_mask
        int[] inputs2; // segment_ids
        float[] outputs0; // end_logits
        float[] outputs1; // start_logits

        Dictionary<string, int> vocabularyTable;

        public Bert(string modelPath, string vocabText)
        {
            // NO GPU
            var options = new InterpreterOptions()
            {
                threads = 2
            };
            interpreter = new Interpreter(FileUtil.LoadFile(modelPath), options);
            interpreter.LogIOInfo();

            inputs0 = new int[MAX_SEQ_LENTH];
            inputs1 = new int[MAX_SEQ_LENTH];
            inputs2 = new int[MAX_SEQ_LENTH];
            outputs0 = new float[MAX_SEQ_LENTH];
            outputs1 = new float[MAX_SEQ_LENTH];

            interpreter.AllocateTensors();

            vocabularyTable = LoadVocabularies(vocabText);
        }

        public void Dispose()
        {
            interpreter?.Dispose();
        }

        public void Invoke(string query, string content)
        {

            PreProcess(query, content);

            interpreter.SetInputTensorData(0, inputs0);
            interpreter.SetInputTensorData(1, inputs1);
            interpreter.SetInputTensorData(2, inputs2);

            interpreter.Invoke();

            ResetArray(outputs0);
            ResetArray(outputs1);
            interpreter.GetOutputTensorData(0, outputs0);
            interpreter.GetOutputTensorData(1, outputs1);
        }

        ContentData PreProcess(string query, string content)
        {
            var queryTokens = BertTokenizer.Fulltokenize(query, vocabularyTable)
                                           .Take(MAX_QUERY_LENTH);

            var contentWords = content.Split((char c) => c.IsBertWhiteSpace(), StringSplitOptions.RemoveEmptyEntries);
            var contentTokenIdxToWordIdxMapping = new List<int>();
            var contentTokens = new List<string>();
            for (int i = 0; i < contentWords.Length; i++)
            {
                var wordTokens = BertTokenizer.Fulltokenize(contentWords[i], vocabularyTable);
                foreach (var subToken in wordTokens)
                {
                    contentTokenIdxToWordIdxMapping.Add(i);
                    contentTokens.Add(subToken);
                }
            }

            // -3 accounts for [CLS], [SEP] and [SEP]
            // int maxContentLen = MAX_SEQ_LENTH - queryTokens.Count() - 3;
            //  contentTokens.AddRange(new string[MAX_SEQ_LENTH - queryTokens.Count() - 3 - contentTokens.Count]);

            var tokens = new List<string>(MAX_SEQ_LENTH);
            var segmentIds = new List<Int32>(MAX_SEQ_LENTH);

            // Map token index to original index (in feature.origTokens).
            var tokenIdxToWordIdxMapping = new Dictionary<int, int>();

            // Start of generating the `InputFeatures`.
            tokens.Add("[CLS]");
            segmentIds.Add(0);

            // For query input.
            foreach (string t in queryTokens)
            {
                tokens.Add(t);
                segmentIds.Add(0);
            }

            // For separation.
            tokens.Add("[SEP]");
            segmentIds.Add(0);

            // For text input.
            for (int i = 0; i < contentTokens.Count; i++)
            {
                tokens.Add(contentTokens[i]);
                segmentIds.Add(1);
                tokenIdxToWordIdxMapping[tokens.Count] = contentTokenIdxToWordIdxMapping[i];
            }

            // For ending mark.
            tokens.Add("[SEP]");
            segmentIds.Add(1);

            ResetArray(inputs0);
            ResetArray(inputs1);
            ResetArray(inputs2);
            for (int i = 0; i < tokens.Count; i++)
            {
                // Input IDs
                inputs0[i] = vocabularyTable[tokens[i]];
                // Input Mask
                inputs1[i] = 1;
                // Segment IDs
                inputs2[i] = segmentIds[i];
            }

            return new ContentData()
            {
                contentWords = contentWords,
                tokenIdxToWordIdxMapping = tokenIdxToWordIdxMapping,
                originalContent = content,
            };
        }

        public Answer[] GetResult()
        {
            // Name Alias
            float[] startLogits = outputs1;
            float[] endLogits = outputs0;

            // Get the candidate start/end indexes of answer from `startLogits` and `endLogits`.
            int[] startIndexes = CandidateAnswerIndexes(startLogits, 5);
            int[] endIndexes = CandidateAnswerIndexes(endLogits, 5);

            // Make list which stores prediction and its range to find original results and filter invalid pairs.     
            var candidates = startIndexes.SelectMany((startIndex) =>
            {
                return endIndexes.Select((endIndex) =>
                {
                    return new Score()
                    {
                        logit = startLogits[startIndex] + endLogits[endIndex],
                        start = startIndex,
                        end = endIndex,
                    };
                });
            }).OrderByDescending(score => score.logit);

            // 

            return null;
        }

        public static Dictionary<string, int> LoadVocabularies(string text)
        {
            var lines = text.Split(new char[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            var vocablaries = new Dictionary<string, int>();
            for (int i = 0; i < lines.Length; i++)
            {
                vocablaries.Add(lines[i], i);
            }
            return vocablaries;
        }

        private static void ResetArray<T>(T[] arr)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] = default(T);
            }
        }

        private static int[] CandidateAnswerIndexes(float[] logits, int answerCount)
        {
            return logits.ToIndexValueTuple()
                .OrderByDescending(t => t.Item2)
                .Take(answerCount)
                .Select(t => t.Item1)
                .ToArray();
        }


    }
}
