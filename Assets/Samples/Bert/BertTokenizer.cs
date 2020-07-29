using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;


namespace TensorFlowLite
{
    // See
    // https://unicode.org/reports/tr44/#General_Category_Values
    // https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/tokenization.py#L161).
    public class BertTokenizer
    {
        public static string[] Fulltokenize(string text, Dictionary<string, int> table)
        {
            return BasicTokenize(text).SelectMany(word =>
            {
                return WordPieceTokenize(word, table);
            }).ToArray();
        }


        public static string[] BasicTokenize(string text)
        {
            var sb = new StringBuilder();
            text = text.Normalize(NormalizationForm.FormC);

            foreach (char c in text)
            {
                if (c.IsBertWhiteSpace())
                {
                    sb.Append(' ');
                    continue;
                }
                if (c.IsBertShouldBeRemoved() || c.IsBertControl())
                {
                    continue;
                }
                sb.Append(c);
            }
            text = sb.ToString().ToLower();
            return text.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries)
                       .SelectMany((word) => TokenizeWithPunctuation(word)).ToArray();
        }

        /// <summary>
        /// Tokenizes a piece of text into its word pieces.
        /// This uses a greedy longest-match-first algorithm to perform tokenization
        /// using the given vocabulary.
        /// </summary>
        /// <param name="text">A single token or whitespace separated tokens. This should have already been passed through `BasicTokenizer.</param>
        /// <param name="table">A Vocabulary Table</param>
        /// <returns>A list of wordpiece tokens.</returns>
        public static string[] WordPieceTokenize(string text, Dictionary<string, int> table)
        {
            const string UNKNOWN_TOKEN = "[UNK]";
            const uint MAX_INPUT_CHARS_PER_WORD = 200;

            var tokens = new List<string>();
            var subTokens = new List<String>();

            var words = text.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            foreach (var word in words)
            {
                if (word.Length >= MAX_INPUT_CHARS_PER_WORD)
                {
                    tokens.Append(UNKNOWN_TOKEN);
                    continue;
                }

                bool isBad = false;
                int start = 0;
                subTokens.Clear();

                while (start < word.Length)
                {
                    int end = word.Length;
                    string curSubstr = null;

                    while (start < end)
                    {
                        string substr = word.Substring(start, end - start);

                        if (start > 0)
                        {
                            substr = "##" + substr;
                        }
                        // UnityEngine.Debug.Log($"s: {start}, e: {end}, substr: {substr}");

                        if (table.ContainsKey(substr))
                        {
                            // UnityEngine.Debug.LogWarning($"found: {substr}");
                            curSubstr = substr;
                            break;
                        }
                        end -= 1;
                    }

                    if (curSubstr == null)
                    {
                        isBad = true;
                        break;
                    }

                    subTokens.Add(curSubstr);
                    start = end;
                }
                if (isBad)
                {
                    tokens.Add(UNKNOWN_TOKEN);
                }
                else
                {
                    tokens.AddRange(subTokens);
                }
            }
            return tokens.ToArray();
        }

        private static string[] TokenizeWithPunctuation(string text)
        {
            var sb = new StringBuilder();
            var tokens = new List<string>();

            foreach (var c in text)
            {
                if (c.IsBertPunctuation())
                {
                    if (sb.Length > 0)
                    {
                        tokens.Add(sb.ToString());
                    }
                    tokens.Add(c.ToString());
                    sb.Clear();
                }
                else
                {
                    sb.Append(c);
                }
            }
            if (sb.Length > 0)
            {
                tokens.Add(sb.ToString());
            }
            return tokens.ToArray();
        }

    }

    public static class CharExtension
    {
        public static bool IsBertWhiteSpace(this char c)
        {
            switch (c)
            {
                case ' ': return true;
                case '\t': return true;
                case '\n': return true;
                case '\r': return true;
            }
            if (CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.SpaceSeparator) return true;
            return false;
        }

        public static bool IsBertControl(this char c)
        {
            if (c.IsBertWhiteSpace()) return false;
            switch (CharUnicodeInfo.GetUnicodeCategory(c))
            {
                case UnicodeCategory.Control: return true;
                case UnicodeCategory.Format: return true;
            }
            return false;
        }

        public static bool IsBertShouldBeRemoved(this char c)
        {
            return c == '\u0000' || c == '\ufffd';
        }

        public static bool IsBertPunctuation(this char c)
        {
            // We treat all non-letter/number ASCII as punctuation.
            // Characters such as "^", "$", and "`" are not in the Unicode
            // Punctuation class but we treat them as punctuation anyways, for consistency.
            if ((c >= 33 && c <= 47)
            || (c >= 58 && c <= 64)
            || (c >= 91 && c <= 96)
            || (c >= 123 && c <= 126))
            {
                return true;
            }
            switch (CharUnicodeInfo.GetUnicodeCategory(c))
            {
                case UnicodeCategory.ConnectorPunctuation:
                case UnicodeCategory.DashPunctuation:
                case UnicodeCategory.OpenPunctuation:
                case UnicodeCategory.ClosePunctuation:
                case UnicodeCategory.InitialQuotePunctuation:
                case UnicodeCategory.FinalQuotePunctuation:
                case UnicodeCategory.OtherPunctuation:
                    return true;
            }
            return false;
        }
    }

    public static class StringExtension
    {
        public static bool IsBertControl(this string s)
        {
            if (!Char.IsSurrogate(s, 0))
            {
                return s[0].IsBertControl();
            }
            switch (CharUnicodeInfo.GetUnicodeCategory(s, 0))
            {
                case UnicodeCategory.Control: return true;
                case UnicodeCategory.Format: return true;
            }
            return false;
        }

        public delegate bool SeparatorFunc(char c);

        public static string[] Split(this string text, SeparatorFunc separator, StringSplitOptions options)
        {
            var components = new List<string>();
            var sb = new StringBuilder();

            foreach (var c in text)
            {
                if (separator(c))
                {
                    if (options == StringSplitOptions.None || sb.Length > 0)
                    {
                        components.Add(sb.ToString());
                    }
                    sb.Clear();
                }
                else
                {
                    sb.Append(c);
                }
            }

            if (sb.Length > 0)
            {
                components.Add(sb.ToString());
            }
            return components.ToArray();
        }
    }

}
