using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;


public class MnistSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "mnist.tflite";
    [SerializeField] Text outputTextView = null;
    [SerializeField] ComputeShader compute = null;

    Interpreter interpreter;

    bool isProcessing = false;
    float[,] inputs = new float[28, 28];
    float[] outputs = new float[10];
    ComputeBuffer inputBuffer;

    System.Text.StringBuilder sb = new System.Text.StringBuilder();

    void Start()
    {
        var options = new InterpreterOptions()
        {
            threads = 2,
            useNNAPI = false,
        };
        interpreter = new Interpreter(FileUtil.LoadFile(fileName), options);
        interpreter.ResizeInputTensor(0, new int[] { 1, 28, 28, 1 });
        interpreter.AllocateTensors();

        inputBuffer = new ComputeBuffer(28 * 28, sizeof(float));
    }

    void OnDestroy()
    {
        interpreter?.Dispose();
        inputBuffer?.Dispose();
    }

    public void OnDrawTexture(RenderTexture texture)
    {
        if (!isProcessing)
        {
            Invoke(texture);
        }
    }

    void Invoke(RenderTexture texture)
    {
        isProcessing = true;

        compute.SetTexture(0, "InputTexture", texture);
        compute.SetBuffer(0, "OutputTensor", inputBuffer);
        compute.Dispatch(0, 28 / 4, 28 / 4, 1);
        inputBuffer.GetData(inputs);

        float startTime = Time.realtimeSinceStartup;
        interpreter.SetInputTensorData(0, inputs);
        interpreter.Invoke();
        interpreter.GetOutputTensorData(0, outputs);
        float duration = Time.realtimeSinceStartup - startTime;

        sb.Clear();
        sb.AppendLine($"Process time: {duration: 0.00000} sec");
        sb.AppendLine("---");
        for (int i = 0; i < outputs.Length; i++)
        {
            sb.AppendLine($"{i}: {outputs[i]: 0.00}");
        }
        outputTextView.text = sb.ToString();

        isProcessing = false;
    }
}
