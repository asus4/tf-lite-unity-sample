using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;

public class BertSample : MonoBehaviour
{
    [System.Serializable]
    public class QASet
    {
        public string title;
        public string content;
        public string[] questions;
    }

    [System.Serializable]
    public class QASetCollection
    {
        public QASet[] contents;
    }

    [Header("TFLite")]
    [SerializeField, FilePopup("*.tflite")] string fileName = "deeplabv3_257_mv_gpu.tflite";
    [SerializeField] TextAsset qaJson = null;
    [SerializeField] TextAsset vocabTable = null;

    [Header("UIs")]
    [SerializeField] Dropdown sentenceDropdown = null;
    [SerializeField] Text sentenceLabel = null;
    [SerializeField] Dropdown templetesDropdown = null;
    [SerializeField] InputField questionInput = null;
    [SerializeField] Button askButton = null;

    [Header("Data Set")]
    [SerializeField] QASet[] dataSets;

    Bert bert;

    QASet CurrentQASet => dataSets[sentenceDropdown.value];

    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        bert = new Bert(path, vocabTable.text);

        dataSets = JsonUtility.FromJson<QASetCollection>(qaJson.text).contents;

        // Init UIs
        sentenceDropdown.ClearOptions();
        sentenceDropdown.AddOptions(dataSets.Select(o => o.title).ToList());
        sentenceDropdown.value = 0;
        sentenceDropdown.onValueChanged.AddListener((value) =>
        {
            Debug.Log($"sentence dropdown: {value}");
            SelectData(dataSets[value]);
        });
        SelectData(dataSets[0]);


        templetesDropdown.onValueChanged.AddListener((value) =>
        {
            questionInput.text = templetesDropdown.captionText.text;
        });
        askButton.onClick.AddListener(() =>
        {
            var question = questionInput.text;
            if (string.IsNullOrWhiteSpace(question)) return;
            Invoke(CurrentQASet, question);
        });
    }

    void OnDestroy()
    {
        sentenceDropdown.onValueChanged.RemoveAllListeners();
        templetesDropdown.onValueChanged.RemoveAllListeners();
        askButton.onClick.RemoveAllListeners();
        bert?.Dispose();
    }

    void SelectData(QASet qa)
    {
        sentenceLabel.text = qa.content;
        templetesDropdown.ClearOptions();
        templetesDropdown.AddOptions(qa.questions.ToList());
        questionInput.text = "";
    }

    void Invoke(QASet qa, string question)
    {
        Debug.Log($"questions: {question}");
        var answers = bert.Invoke(question, qa.content);
        if (answers.Length == 0)
        {
            Debug.LogError("Answer Not Found!");
            return;
        }
        for (int i = 0; i < answers.Length; i++)
        {
            Debug.Log($"Answer {i}: {answers[i]}");
        }

        sentenceLabel.text = GenerateHighlightedText(qa.content, answers.First());
    }

    string GenerateHighlightedText(string text, Bert.Answer answer)
    {
        var match = answer.matched;
        return text.Substring(0, match.Index)
            + "<b><color=#ffa500ff>"
            + match.Value
            + "</color></b>"
            + text.Substring(match.Index + match.Length);
    }

}
