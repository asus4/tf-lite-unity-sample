using System.Collections;
using System.Diagnostics;
using System.Text;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;
using Debug = UnityEngine.Debug;

public class GpuBindSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "meet/segm_lite_v509_128x128_float16_quant.tflite";
    [SerializeField] Texture2D inputTex = null;
    [SerializeField] RawImage outputImage = null;
    [SerializeField] bool useBinding = false;
    [SerializeField] ComputeShader computeNormal = null;
    [SerializeField] ComputeShader computePadded = null;

    static readonly double msec = 1000.0 / Stopwatch.Frequency;
    Stopwatch stopwatch;
    RenderTexture outputTex;
    Color32[] textureBuffer;

    IEnumerator Start()
    {
        Debug.Assert(IsGpuBindingSupported);

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
        ComputeBuffer inputBuffer = null, outputBuffer = null;

        // Manage gpu delegate manualy
        using (IBindableDelegate gpuDelegate = CreateGpuDelegate(useBinding))
        using (Interpreter interpreter = new Interpreter(FileUtil.LoadFile(fileName), new InterpreterOptions()))
        {
            bool isMetal = Application.platform != RuntimePlatform.Android;
            if (!useBinding || isMetal)
            {
                if (interpreter.ModifyGraphWithDelegate(gpuDelegate) != Interpreter.Status.Ok)
                {
                    Debug.LogError("Failed to modify the graph with delegate");
                }
            }
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
                // On iOS GPU, input must be 4 channels, regardless of what model expects.
                int gpuInputChannels = isMetal ? 4 : 3;
                int gpuOutputChannels = isMetal ? 4 : 2;

                inputBuffer = new ComputeBuffer(height * width * gpuInputChannels, sizeof(float), ComputeBufferType.Structured);
                float[,,] inputs = new float[height, width, gpuInputChannels];
                TextureToTensor(inputTex, inputs);
                inputBuffer.SetData(inputs);
                if (!gpuDelegate.BindBufferToInputTensor(interpreter, 0, inputBuffer))
                {
                    Debug.LogError("input is not binded");
                }
                Debug.Log($"input size: {inputBuffer.count} channels:{gpuInputChannels}");

                outputBuffer = new ComputeBuffer(height * width * gpuOutputChannels, sizeof(float), ComputeBufferType.Structured);
                outputBuffer.SetData(outputs);
                interpreter.SetAllowBufferHandleOutput(true);
                if (!gpuDelegate.BindBufferToOutputTensor(interpreter, 0, outputBuffer))
                {
                    Debug.LogError("output is not binded");
                }
                Debug.Log($"output size: {outputBuffer.count} channels:{gpuOutputChannels}");

                if (!isMetal)
                {
                    if (interpreter.ModifyGraphWithDelegate(gpuDelegate) != Interpreter.Status.Ok)
                    {
                        Debug.LogError("Failed to modify the graph with delegate");
                    }
                    Debug.Log("modified android graph");
                }
            }
            else
            {
                float[,,] inputs = new float[height, width, channels];
                TextureToTensor(inputTex, inputs);
                interpreter.SetInputTensorData(0, inputs);

                outputBuffer = new ComputeBuffer(height * width * 2, sizeof(float));
            }
            StopSW(sb, "Prepare inputs/outputs");

            // Invoke
            StartSW();
            interpreter.Invoke();
            StopSW(sb, "Invoke");

            StartSW();
            if (!useBinding)
            {
                interpreter.GetOutputTensorData(0, outputs);
                outputBuffer.SetData(outputs);
            }
            var compute = (isMetal && useBinding) ? computePadded : computeNormal;
            // var compute = (useBinding) ? computePadded : computeNormal;

            RenderToOutputTexture(compute, outputBuffer, outputTex);

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

    static void TextureToTensor(Texture2D texture, float[,,] tensor)
    {
        Color32[] pixels = texture.GetPixels32();
        int width = texture.width;
        int height = texture.height - 1;
        const float scale = 255f;
        int channels = tensor.GetLength(2);

        if (channels == 3)
        {
            for (int i = 0; i < pixels.Length; i++)
            {
                int y = height - i / width;
                int x = i % width;
                tensor[y, x, 0] = (float)(pixels[i].r) / scale;
                tensor[y, x, 1] = (float)(pixels[i].g) / scale;
                tensor[y, x, 2] = (float)(pixels[i].b) / scale;
            }
        }
        else if (channels == 4)
        {
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
        else if (channels > 4)
        {
            for (int i = 0; i < pixels.Length; i++)
            {
                int y = height - i / width;
                int x = i % width;
                tensor[y, x, 0] = (float)(pixels[i].r) / scale;
                tensor[y, x, 1] = (float)(pixels[i].g) / scale;
                tensor[y, x, 2] = (float)(pixels[i].b) / scale;
                for (int c = 3; c < channels; c++)
                {
                    tensor[y, x, c] = 1f;
                }
            }
        }
        else
        {
            throw new System.NotSupportedException();
        }
    }

    static bool IsGpuBindingSupported
    {
        get
        {
            switch (SystemInfo.graphicsDeviceType)
            {
                case GraphicsDeviceType.Metal:
                case GraphicsDeviceType.OpenGLES3:
                    return true;
            }
            return false;
        }
    }

#pragma warning disable CS0162 // Unreachable code detected 
    static IBindableDelegate CreateGpuDelegate(bool useBinding)
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        var glOptions = GpuDelegateV2.DefaultOptions;
        if (useBinding)
        {
            glOptions.isPrecisionLossAllowed = 1;
            glOptions.inferencePreference = (int)GpuDelegateV2.Usage.SustainedSpeed;
            glOptions.inferencePriority1 = (int)GpuDelegateV2.InferencePriority.MinLatency;
            glOptions.inferencePriority2 = (int)GpuDelegateV2.InferencePriority.Auto;
            glOptions.inferencePriority3 = (int)GpuDelegateV2.InferencePriority.Auto;
        }
        return new GpuDelegateV2(glOptions);
#elif UNITY_IOS || UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
        return new MetalDelegate(new MetalDelegate.Options()
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
#endif
        throw new System.NotSupportedException();
        return null;
    }
#pragma warning restore CS0162 // Unreachable code detected    

}
