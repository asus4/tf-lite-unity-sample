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
    [SerializeField, FilePopup("*.tflite")] string fileName = "meet/segm_lite_v509_128x128_float16_quant.tflite";
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
            // waitType = MetalDelegate.WaitType.Passive,
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

            ComputeBuffer inputBuffer = null, outputBuffer = null;
            float[,,] outputs;
            if (useBinding)
            {
                int inputTensorIndex0 = interpreter.GetInputTensorIndex(0);
                int outputTensorIndex0 = interpreter.GetOutputTensorIndex(0);
                sb.AppendLine($"Tensor Index = in0:{inputTensorIndex0} out0:{outputTensorIndex0}");

                // On iOS GPU, input must be 4 channels, regardless of what model expects.
                inputBuffer = new ComputeBuffer(height * width * 4, sizeof(float));
                float[,,] inputs = new float[height, width, 4];
                TextureToTensorRGBA(inputTex, inputs);
                inputBuffer.SetData(inputs);
                if (!metalDelegate.BindBufferToTensor(inputTensorIndex0, inputBuffer))
                {
                    Debug.LogError("input is not binded");
                }

                outputBuffer = new ComputeBuffer(height * width * 2, sizeof(float));
                outputs = new float[height, width, 2];
                outputBuffer.SetData(outputs);
                interpreter.SetAllowBufferHandleOutput(true);

                if (!metalDelegate.BindBufferToTensor(outputTensorIndex0, outputBuffer))
                {
                    Debug.LogError("output is not binded");
                }
            }
            else
            {
                float[,,] inputs = new float[height, width, channels];
                TextureToTensorRGB(inputTex, inputs);
                interpreter.SetInputTensorData(0, inputs);

                outputs = new float[height, width, 2];
            }
            StopSW(sb, "Prepare inputs/outputs");

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
            // Debug.Assert(ValidateTensor(outputs));

            StopSW(sb, "Get Output");
            StartSW();
            // DeepLab.ResultToTexture2D(outputs, outputTex, textureBuffer);
            StopSW(sb, "Post Process");

            // Cleanup
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

    static void TextureToTensorRGB(Texture2D texture, float[,,] tensor)
    {
        Debug.Assert(tensor.GetLength(2) == 3);
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

    static void TextureToTensorRGBA(Texture2D texture, float[,,] tensor)
    {
        Debug.Assert(tensor.GetLength(2) == 4);
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
            tensor[y, x, 3] = 1f;
        }
    }

    static bool ValidateTensor(float[,,] tensor)
    {
        int rows = tensor.GetLength(0); // y
        int cols = tensor.GetLength(1); // x
        int labels = tensor.GetLength(2);

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                for (int n = 0; n < labels; n++)
                {
                    if (tensor[y, x, n] != 0)
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }

}
