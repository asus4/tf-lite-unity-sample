using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;
using Cysharp.Threading.Tasks;

/// <summary>
/// Mobile BERT Question Answering Sample
/// 
/// https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/
/// </summary>
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
    [SerializeField]
    private RemoteFile modelFile = new("https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/bert_qa/ios/models_tflite_bert_qa_mobilebert_float_20191023.tflite");

    [SerializeField] TextAsset qaJson = null;
    [SerializeField] TextAsset vocabTable = null;

    [Header("UIs")]
    [SerializeField] Dropdown sentenceDropdown = null;
    [SerializeField] Text sentenceLabel = null;
    [SerializeField] Dropdown templatesDropdown = null;
    [SerializeField] InputField questionInput = null;
    [SerializeField] Button askButton = null;

    [Header("Data Set")]
    [SerializeField] QASet[] dataSets;

    Bert bert;

    QASet CurrentQASet => dataSets[sentenceDropdown.value];

    async void Start()
    {
        sentenceLabel.text = "NOW Loading...";

        // Load model file asynchronously
        var modelData = await modelFile.Load(destroyCancellationToken);
        bert = new Bert(modelData, vocabTable.text);

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

        templatesDropdown.onValueChanged.AddListener((value) =>
        {
            questionInput.text = templatesDropdown.captionText.text;
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
        templatesDropdown.onValueChanged.RemoveAllListeners();
        askButton.onClick.RemoveAllListeners();
        bert?.Dispose();
    }

    void SelectData(QASet qa)
    {
        sentenceLabel.text = qa.content;
        templatesDropdown.ClearOptions();
        templatesDropdown.AddOptions(qa.questions.ToList());
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
        return text[..match.Index]
            + "<b><color=#ffa500ff>"
            + match.Value
            + "</color></b>"
            + text[(match.Index + match.Length)..];
    }

}
