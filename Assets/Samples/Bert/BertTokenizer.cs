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

        public static string[] TokenizeWithPunctuation(string text)
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

}
