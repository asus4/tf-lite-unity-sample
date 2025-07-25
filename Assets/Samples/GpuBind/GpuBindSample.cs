using System.Collections;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
using UnityEngine.UI;
using Debug = UnityEngine.Debug;

/// <summary>
/// A demo for GPU binding to accelerate CPU <-> GPU data transfer.
/// 
/// See issue for more details:
/// https://github.com/asus4/tf-lite-unity-sample/issues/23
/// </summary>
public sealed class GpuBindSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "meet/segm_lite_v509_128x128_float16_quant.tflite";
    [SerializeField] Texture2D inputTex = null;
    [SerializeField] RawImage outputImage = null;
    [SerializeField] Text infoLabel = null;

    [SerializeField] bool useBinding = false;

    [SerializeField] ComputeShader computePreProcess = null;
    [SerializeField] ComputeShader computePostProcess = null;

    const string USE_PADDED_KEYWORD = "USE_PADDED";
    static bool IsMetal => SystemInfo.graphicsDeviceType == GraphicsDeviceType.Metal;
    static bool IsOpenGLES3 => SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLES3;

    IBindableDelegate gpuDelegate;
    Interpreter interpreter;

    CommandBuffer commandBuffer;
    float[,,] inputs;
    float[,,] outputs;
    ComputeBuffer inputBuffer;
    ComputeBuffer outputBuffer;
    RenderTexture outputTex;
    bool isReady = false;

    IEnumerator Start()
    {
        Debug.Assert(IsGpuBindingSupported);

        infoLabel.text = useBinding ? "Binding: On" : "Binding Off";

        outputTex = new RenderTexture(inputTex.width, inputTex.height, 0, RenderTextureFormat.ARGB32);
        outputTex.enableRandomWrite = true;
        outputTex.Create();
        outputImage.texture = outputTex;

        commandBuffer = new CommandBuffer()
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

        yield return new WaitForEndOfFrame();
        yield return new WaitForEndOfFrame();
        isReady = true;
    }

    private void OnDestroy()
    {
        StopAllCoroutines();

        interpreter?.Dispose();
        if (!IsMetal)
        {
            gpuDelegate?.Dispose();
        }

        inputBuffer?.Release();
        outputBuffer?.Release();

        commandBuffer?.Release();

        if (outputTex != null)
        {
            Destroy(outputTex);
        }
    }

    void Update()
    {
        if (!isReady) return;

        if (useBinding)
        {
            InvokeBindingOn(inputTex);
        }
        else
        {
            InvokeBindingOff(inputTex);
        }
    }


    void PrepareBindingOn()
    {
        bool isMetal = IsMetal;

        // Set shader keywords based on platform
        if (isMetal)
        {
            computePreProcess.EnableKeyword(USE_PADDED_KEYWORD);
            computePostProcess.EnableKeyword(USE_PADDED_KEYWORD);
        }
        else
        {
            computePreProcess.DisableKeyword(USE_PADDED_KEYWORD);
            computePostProcess.DisableKeyword(USE_PADDED_KEYWORD);
        }

        gpuDelegate = CreateGpuDelegate(true);
        var options = new InterpreterOptions();
        // [Metal] must be called ModifyGraphWithDelegate at beginning
        if (isMetal)
        {
            options.AddDelegate(gpuDelegate);
        }
        interpreter = new Interpreter(FileUtil.LoadFile(fileName), options);



        var inputShape0 = interpreter.GetInputTensorInfo(0).shape;
        int height = inputShape0[1];
        int width = inputShape0[2];
        // int channels = inputShape0[3];

        // On MetalDelegate, channels must be 4 padded, regardless of the model I/O.
        int gpuInputChannels = isMetal ? 4 : 3;
        int gpuOutputChannels = isMetal ? 4 : 2;

        inputBuffer = new ComputeBuffer(height * width * gpuInputChannels, sizeof(float));
        inputs = new float[height, width, gpuInputChannels];
        if (!gpuDelegate.BindBufferToInputTensor(interpreter, 0, inputBuffer))
        {
            Debug.LogError("input is not bound");
        }

        outputBuffer = new ComputeBuffer(height * width * gpuOutputChannels, sizeof(float));
        interpreter.SetAllowBufferHandleOutput(true);
        if (!gpuDelegate.BindBufferToOutputTensor(interpreter, 0, outputBuffer))
        {
            Debug.LogError("output is not bound");
        }

        // [OpenGLGLES] must be called ModifyGraphWithDelegate at last  
        if (IsOpenGLES3)
        {
            // Gpu Delegate must be initialized in the Render thread to use the same egl context.
            RenderThreadHook.RunOnRenderThread(() =>
            {
                if (interpreter.ModifyGraphWithDelegate(gpuDelegate) != Interpreter.Status.Ok)
                {
                    Debug.LogError("Failed to modify the graph with delegate");
                }
            });
        }
    }

    void PrepareBindingOff()
    {
        // Always use non-padded version when binding is off
        computePreProcess.DisableKeyword(USE_PADDED_KEYWORD);
        computePostProcess.DisableKeyword(USE_PADDED_KEYWORD);

        var options = new InterpreterOptions();
        options.AddGpuDelegate();
        interpreter = new Interpreter(FileUtil.LoadFile(fileName), options);

        var inputShape0 = interpreter.GetInputTensorInfo(0).shape;
        int height = inputShape0[1];
        int width = inputShape0[2];
        int channels = inputShape0[3];

        inputs = new float[height, width, channels];
        outputs = new float[height, width, 2];

        inputBuffer = new ComputeBuffer(height * width * 3, sizeof(float));
        outputBuffer = new ComputeBuffer(height * width * 2, sizeof(float));
    }


    void InvokeBindingOn(Texture2D inputTex)
    {
        TextureToTensor(inputTex, computePreProcess, inputBuffer);

        Profiler.BeginSample("Invoke");
        if (Application.platform == RuntimePlatform.Android)
        {
            // On GpuDelegateV2 (Android), Invoke() must be called in the render thread to use the same egl context.
            RenderThreadHook.RunOnRenderThread(() =>
            {
                interpreter.Invoke();
            });
        }
        else
        {
            // On MetalDelegate (iOS or macOS), it can be called in the main thread.
            interpreter.Invoke();
        }
        Profiler.EndSample();

        Profiler.BeginSample("Post process");
        RenderToOutputTexture(computePostProcess, outputBuffer, outputTex);
        Profiler.EndSample();
    }

    void InvokeBindingOff(Texture2D inputTex)
    {
        TextureToTensor(inputTex, computePreProcess, inputBuffer);
        inputBuffer.GetData(inputs);
        interpreter.SetInputTensorData(0, inputs);

        Profiler.BeginSample("Invoke");
        interpreter.Invoke();
        Profiler.EndSample();

        Profiler.BeginSample("Post process");
        interpreter.GetOutputTensorData(0, outputs);
        outputBuffer.SetData(outputs);
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

        commandBuffer.Clear();
        commandBuffer.SetExecutionFlags(CommandBufferExecutionFlags.None);
        // var fence = commandBuffer.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, SynchronisationStageFlags.ComputeProcessing);

        commandBuffer.SetComputeIntParam(compute, "Width", texture.width);
        commandBuffer.SetComputeIntParam(compute, "Height", texture.height);
        commandBuffer.SetComputeTextureParam(compute, 0, "InputTexture", texture);
        commandBuffer.SetComputeBufferParam(compute, 0, "OutputTensor", tensor);
        commandBuffer.DispatchCompute(compute, 0, texture.width / 8, texture.height / 8, 1);

        Graphics.ExecuteCommandBuffer(commandBuffer);
        // Graphics.WaitOnAsyncGraphicsFence(fence);
    }

    static bool IsGpuBindingSupported => SystemInfo.graphicsDeviceType switch
    {
        GraphicsDeviceType.Metal or GraphicsDeviceType.OpenGLES3 => true,
        _ => false,
    };

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
            waitType = useBinding
                ? MetalDelegate.WaitType.Active
                : MetalDelegate.WaitType.Passive,
            enableQuantization = true,
        });
#endif
        throw new System.NotSupportedException();
    }
}
