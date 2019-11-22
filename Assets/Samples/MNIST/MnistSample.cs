using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;


public class MnistSample : MonoBehaviour
{
    [SerializeField] string fileName = "mnist.tflite";
    [SerializeField] Text outputTextView = null;
    [SerializeField] ComputeShader compute = null;
    [SerializeField] bool useGPUDelegate = false;

    Interpreter interpreter;

    bool isProcessing = false;
    float[,] inputs = new float[28, 28];
    float[] outputs = new float[10];
    ComputeBuffer inputBuffer;

    System.Text.StringBuilder sb = new System.Text.StringBuilder();

    void Start()
    {
        GpuDelegate gpuDelegate = null;
        if (useGPUDelegate)
        {
            gpuDelegate = new MetalDelegate(new MetalDelegate.TFLGpuDelegateOptions()
            {
                allow_precision_loss = false,
                waitType = MetalDelegate.TFLGpuDelegateWaitType.Passive,
            });
        }
        var path = Path.Combine(Application.streamingAssetsPath, fileName);
        interpreter = new Interpreter(File.ReadAllBytes(path), 1, gpuDelegate);
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
            Excecute(texture);
        }
    }

    void Excecute(RenderTexture texture)
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
