using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;
using Debug = UnityEngine.Debug;

public class GpuBindSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "deeplabv3_257_mv_gpu.tflite";
    [SerializeField] Texture2D inputTex = null;
    [SerializeField] RawImage outputImage = null;
    [SerializeField] bool useBinding = false;

    static readonly double msec = 1000.0 / Stopwatch.Frequency;
    Stopwatch stopwatch;
    Texture2D outputTex;
    Color32[] textureBuffer;

    IEnumerator Start()
    {
        stopwatch = new Stopwatch();
        outputTex = new Texture2D(inputTex.width, inputTex.height, TextureFormat.RGBA32, 0, false);
        outputImage.texture = outputTex;
        textureBuffer = new Color32[inputTex.width * inputTex.height];

        // bool useBinding = false;
        for (int i = 0; i < 10; i++)
        {
            yield return new WaitForEndOfFrame();
            RunInterpreter(useBinding);
            // useBinding = !useBinding;
        }
    }

    private void OnDestroy()
    {
        if (outputTex != null)
        {
            Destroy(outputTex);
        }
    }

    void RunInterpreter(bool useBinding)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine(useBinding ? "Binding On" : "Binding Off");

        // Prepare
        StartSW();
        var metalDelegate = new MetalDelegate(new MetalDelegate.Options()
        {
            allowPrecisionLoss = false,
            waitType = useBinding
                ? MetalDelegate.WaitType.Active
                : MetalDelegate.WaitType.Passive,
            enableQuantization = true,
        });
        var options = new InterpreterOptions();
        options.AddGpuDelegate(metalDelegate);

        using (var interpreter = new Interpreter(FileUtil.LoadFile(fileName), options))
        {
            StopSW(sb, "Prepare interpreter");

            // Prepare inputs/outputs
            StartSW();
            var inputShape0 = interpreter.GetInputTensorInfo(0).shape;
            int height = inputShape0[1];
            int width = inputShape0[2];
            int channels = inputShape0[3];

            Debug.Assert(width == inputTex.width, $"{inputTex.width}");
            Debug.Assert(height == inputTex.height, $"{inputTex.height}");

            int inputTensorIndex0 = interpreter.GetInputTensorIndex(0);
            int outputTensorIndex0 = interpreter.GetOutputTensorIndex(0);
            sb.AppendLine($"Tensor Index = in0:{inputTensorIndex0} out0:{outputTensorIndex0}");

            ComputeBuffer inputBuffer = null, outputBuffer = null;
            float[,,] inputs = new float[height, width, channels];
            float[,,] outputs = new float[height, width, 21];
            TextureToTensor(inputTex, inputs);
            if (useBinding)
            {
                inputBuffer = new ComputeBuffer(height * width * channels, sizeof(float));
                inputBuffer.SetData(inputs);

                outputBuffer = new ComputeBuffer(height * width * 21, sizeof(float));
                metalDelegate.BindBufferToTensor(inputTensorIndex0, inputBuffer);
                metalDelegate.BindBufferToTensor(outputTensorIndex0, outputBuffer);
                interpreter.SetAllowBufferHandleOutput(true);
            }
            StopSW(sb, "Prepare inputs/outputs");

            // Set Input
            StartSW();
            if (useBinding)
            {
            }
            else
            {
                interpreter.SetInputTensorData(0, inputs);
            }
            StopSW(sb, "Set Input");

            // Invoke
            StartSW();
            interpreter.Invoke();
            StopSW(sb, "Invoke");

            StartSW();
            if (useBinding)
            {
                outputBuffer.GetData(outputs);
            }
            else
            {
                interpreter.GetOutputTensorData(0, outputs);
            }
            StopSW(sb, "Get Output");
            StartSW();
            DeepLab.ResultToTexture2D(outputs, outputTex, textureBuffer);
            StopSW(sb, "Post Process");
            inputBuffer?.Release();
            inputBuffer?.Dispose();
            outputBuffer?.Release();
            outputBuffer?.Dispose();
        }

        Debug.Log(sb.ToString());
    }

    void StartSW()
    {
        stopwatch.Restart();
    }

    void StopSW(StringBuilder sb, string message)
    {
        stopwatch.Stop();
        sb.AppendLine($"{message}: {stopwatch.ElapsedTicks * msec:0.00} ms");
    }

    static void TextureToTensor(Texture2D texture, float[,,] tensor)
    {
        Color32[] pixels = texture.GetPixels32();
        int width = texture.width;
        int height = texture.height - 1;
        const float scale = 255f;
        for (int i = 0; i < pixels.Length; i++)
        {
            int y = height - i / width;
            int x = i % width;
            tensor[y, x, 0] = (float)(pixels[i].r) / scale;
            tensor[y, x, 1] = (float)(pixels[i].g) / scale;
            tensor[y, x, 2] = (float)(pixels[i].b) / scale;
        }
    }

}
