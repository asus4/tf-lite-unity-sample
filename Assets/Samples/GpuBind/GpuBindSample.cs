using System.Collections;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
using UnityEngine.UI;
using Debug = UnityEngine.Debug;

public class GpuBindSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "meet/segm_lite_v509_128x128_float16_quant.tflite";
    [SerializeField] Texture2D inputTex = null;
    [SerializeField] RawImage outputImage = null;
    [SerializeField] bool useBinding = false;

    [SerializeField] ComputeShader computePreProcessNormal = null;
    [SerializeField] ComputeShader computePreProcessPadded = null;

    [SerializeField] ComputeShader computePostProcessNormal = null;
    [SerializeField] ComputeShader computePostProcessPadded = null;

    static bool IsMetal => SystemInfo.graphicsDeviceType == GraphicsDeviceType.Metal;
    static bool IsOpenGLES3 => SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLES3;


    IBindableDelegate gpuDelegate;
    Interpreter interpreter;

    CommandBuffer preprocessCommand;
    float[,,] inputs;
    float[,,] outputs;
    ComputeBuffer inputBuffer;
    ComputeBuffer outputBuffer;
    RenderTexture outputTex;

    IEnumerator Start()
    {
        Debug.Assert(IsGpuBindingSupported);

        outputTex = new RenderTexture(inputTex.width, inputTex.height, 0, RenderTextureFormat.ARGB32);
        outputTex.enableRandomWrite = true;
        outputTex.Create();
        outputImage.texture = outputTex;

        // Need to wait 1 frame to wait GPU startup
        yield return new WaitForEndOfFrame();

        preprocessCommand = new CommandBuffer()
        {
            name = "preprocess",
        };

        if (useBinding)
        {
            PrepareBindingOn();
        }
        else
        {
            PrepareBindingOff();
        }

        StartCoroutine(Invoke(inputTex, useBinding));
    }

    private void OnDestroy()
    {
        interpreter?.Dispose();
        gpuDelegate?.Dispose();

        inputBuffer?.Release();
        outputBuffer?.Release();

        preprocessCommand?.Release();

        if (outputTex != null)
        {
            Destroy(outputTex);
        }
    }

    void PrepareBindingOn()
    {
        gpuDelegate = CreateGpuDelegate(true);
        interpreter = new Interpreter(FileUtil.LoadFile(fileName), new InterpreterOptions());

        bool isMetal = IsMetal;
        // Metal: Call ModifyGraphWithDelegate at beginning
        if (isMetal)
        {
            if (interpreter.ModifyGraphWithDelegate(gpuDelegate) != Interpreter.Status.Ok)
            {
                Debug.LogError("Failed to modify the graph with delegate");
            }
        }

        var inputShape0 = interpreter.GetInputTensorInfo(0).shape;
        int height = inputShape0[1];
        int width = inputShape0[2];
        // int channels = inputShape0[3];

        // On iOS GPU, input must be 4 channels, regardless of what model expects.
        int gpuInputChannels = isMetal ? 4 : 3;
        int gpuOutputChannels = isMetal ? 4 : 2;

        inputBuffer = new ComputeBuffer(height * width * gpuInputChannels, sizeof(float));
        inputs = new float[height, width, gpuInputChannels];
        if (!gpuDelegate.BindBufferToInputTensor(interpreter, 0, inputBuffer))
        {
            Debug.LogError("input is not binded");
        }

        outputBuffer = new ComputeBuffer(height * width * gpuOutputChannels, sizeof(float));
        interpreter.SetAllowBufferHandleOutput(true);
        if (!gpuDelegate.BindBufferToOutputTensor(interpreter, 0, outputBuffer))
        {
            Debug.LogError("output is not binded");
        }

        // GLES3: Call ModifyGraphWithDelegate at last  
        if (IsOpenGLES3)
        {
            if (interpreter.ModifyGraphWithDelegate(gpuDelegate) != Interpreter.Status.Ok)
            {
                Debug.LogError("Failed to modify the graph with delegate");
            }
        }
    }

    void PrepareBindingOff()
    {
        var options = new InterpreterOptions();
        options.AddGpuDelegate();
        interpreter = new Interpreter(FileUtil.LoadFile(fileName), options);

        var inputShape0 = interpreter.GetInputTensorInfo(0).shape;
        int height = inputShape0[1];
        int width = inputShape0[2];
        int channels = inputShape0[3];

        inputs = new float[height, width, channels];
        outputs = new float[height, width, 2];

        outputBuffer = new ComputeBuffer(height * width * 2, sizeof(float));
    }

    public bool waitPreprocess = true;
    IEnumerator Invoke(Texture2D inputTex, bool useBinding)
    {
        CommandBuffer cmd = new CommandBuffer();
        bool usePadded = IsMetal && useBinding;
        Debug.Log($"usePadded: {usePadded}");

        if (useBinding)
        {
            var computePreProcess = usePadded ? computePreProcessPadded : computePreProcessNormal;
            TextureToTensor(inputTex, computePreProcess, inputBuffer);

            if (waitPreprocess)
            {
                yield return new WaitForFixedUpdate();
            }
            // TextureToTensor(inputTex, inputs);
            // inputBuffer.SetData(inputs);
        }
        else
        {
            TextureToTensor(inputTex, inputs);
            interpreter.SetInputTensorData(0, inputs);
        }

        Profiler.BeginSample("Invoke");
        interpreter.Invoke();
        Profiler.EndSample();

        Profiler.BeginSample("Post process");
        if (!useBinding)
        {
            interpreter.GetOutputTensorData(0, outputs);
            outputBuffer.SetData(outputs);
        }
        var computePostProcess = usePadded ? computePostProcessPadded : computePostProcessNormal;
        RenderToOutputTexture(computePostProcess, outputBuffer, outputTex);
        Profiler.EndSample();
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

    void TextureToTensor(Texture2D texture, ComputeShader compute, ComputeBuffer tensor)
    {
        Debug.Assert(texture.width % 8 == 0);
        Debug.Assert(texture.height % 8 == 0);

        // compute.SetInt("Width", texture.width);
        // compute.SetInt("Height", texture.height);
        // compute.SetTexture(0, "InputTexture", texture);
        // compute.SetBuffer(0, "OutputTensor", tensor);
        // compute.Dispatch(0, texture.width / 8, texture.height / 8, 1);

        preprocessCommand.Clear();
        preprocessCommand.SetExecutionFlags(CommandBufferExecutionFlags.None);
        var fence = preprocessCommand.CreateGraphicsFence(GraphicsFenceType.CPUSynchronisation, SynchronisationStageFlags.AllGPUOperations);

        preprocessCommand.SetComputeIntParam(compute, "Width", texture.width);
        preprocessCommand.SetComputeIntParam(compute, "Height", texture.height);
        preprocessCommand.SetComputeTextureParam(compute, 0, "InputTexture", texture);
        preprocessCommand.SetComputeBufferParam(compute, 0, "OutputTensor", tensor);
        preprocessCommand.DispatchCompute(compute, 0, texture.width / 8, texture.height / 8, 1);

        // Graphics.ExecuteCommandBufferAsync(preprocessCommand, ComputeQueueType.Urgent);
        Graphics.ExecuteCommandBuffer(preprocessCommand);
        Graphics.WaitOnAsyncGraphicsFence(fence);
        Debug.Log($"supportCraphicsFence: {SystemInfo.supportsGraphicsFence}, supportsAsyncCompute: {SystemInfo.supportsAsyncCompute}");

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
