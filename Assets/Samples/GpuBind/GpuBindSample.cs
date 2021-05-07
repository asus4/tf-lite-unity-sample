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
    [SerializeField] ComputeShader computeCPU = null;
    [SerializeField] ComputeShader computeGPU = null;

    static readonly double msec = 1000.0 / Stopwatch.Frequency;
    Stopwatch stopwatch;
    RenderTexture outputTex;
    Color32[] textureBuffer;

    IEnumerator Start()
    {
        stopwatch = new Stopwatch();
        outputTex = new RenderTexture(inputTex.width, inputTex.height, 0, RenderTextureFormat.ARGB32);
        outputTex.enableRandomWrite = true;
        outputTex.Create();
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
            // WaitType.Active might be broke Unity Editor
            // So it is enabled only in iOS
            waitType = useBinding && Application.platform == RuntimePlatform.IPhonePlayer
                ? MetalDelegate.WaitType.Active
                : MetalDelegate.WaitType.Passive,
            enableQuantization = true,
        });
        var options = new InterpreterOptions();
        options.AddGpuDelegate(metalDelegate);


        ComputeBuffer inputBuffer = null, outputBuffer = null;
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

            float[,,] outputs = new float[height, width, 2];

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

                // The buffer size is modified to the next multiple of 4 
                // https://github.com/google/mediapipe/blob/ecb5b5f44ab23ea620ef97a479407c699e424aa7/mediapipe/calculators/tflite/tflite_inference_calculator.cc#L1046-L1047
                outputBuffer = new ComputeBuffer(height * width * 4, sizeof(float));
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

                outputBuffer = new ComputeBuffer(height * width * 2, sizeof(float));
            }
            StopSW(sb, "Prepare inputs/outputs");

            // Invoke
            StartSW();
            interpreter.Invoke();
            StopSW(sb, "Invoke");

            StartSW();
            if (useBinding)
            {
                RenderToOutputTexture(computeGPU, outputBuffer, outputTex);
            }
            else
            {
                interpreter.GetOutputTensorData(0, outputs);
                outputBuffer.SetData(outputs);
                RenderToOutputTexture(computeCPU, outputBuffer, outputTex);
            }
            StopSW(sb, "Post Process");
        }

        // Cleanup
        inputBuffer?.Release();
        inputBuffer?.Dispose();
        outputBuffer?.Release();
        outputBuffer?.Dispose();

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

    static void RenderToOutputTexture(ComputeShader compute, ComputeBuffer buffer, RenderTexture texture)
    {
        Debug.Assert(texture.width % 8 == 0);
        Debug.Assert(texture.height % 8 == 0);

        compute.SetInt("Width", texture.width);
        compute.SetInt("Height", texture.height);
        compute.SetBuffer(0, "LabelBuffer", buffer);
        compute.SetTexture(0, "Result", texture);
        compute.Dispatch(0, texture.width / 8, texture.height / 8, 1);
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

}
