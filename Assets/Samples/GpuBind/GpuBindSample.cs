using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using TensorFlowLite;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class GpuBindSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "deeplabv3_257_mv_gpu.tflite";

    static readonly double msec = 1000.0 / Stopwatch.Frequency;
    Stopwatch stopwatch;

    IEnumerator Start()
    {
        stopwatch = new Stopwatch();

        bool useBinding = false;
        for (int i = 0; i < 10; i++)
        {
            yield return new WaitForEndOfFrame();
            RunInterpreter(useBinding);
            useBinding = !useBinding;
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
            waitType = MetalDelegate.WaitType.Passive,
            enableQuantization = true,
        });
        var options = new InterpreterOptions();
        options.AddGpuDelegate(metalDelegate);

        using (var interpreter = new Interpreter(FileUtil.LoadFile(fileName), options))
        {
            StopSW(sb, "Prepere interpreter");

            // Set input
            StartSW();
            var inputShape0 = interpreter.GetInputTensorInfo(0).shape;
            int height = inputShape0[1];
            int width = inputShape0[2];
            int channels = inputShape0[3];

            ComputeBuffer inputBuffer = null;
            if (useBinding)
            {
                inputBuffer = new ComputeBuffer(height * width * channels, sizeof(float));
                metalDelegate.BindBufferToTensor(0, inputBuffer);
                interpreter.SetAllowBufferHandleOutput(true);
            }
            else
            {
                var inputs = new float[height * width * channels];
                interpreter.SetInputTensorData(0, inputs);
            }
            StopSW(sb, "Set input");
            StartSW();

            interpreter.Invoke();
            StopSW(sb, "Invoke");


            inputBuffer?.Release();
            inputBuffer?.Dispose();
        }

        Debug.Log(sb.ToString());
    }

    void StartSW()
    {
        stopwatch.Restart();
    }

    void StopSW(StringBuilder sb, string message)
    {
        sb.AppendLine($"{message}: {stopwatch.ElapsedTicks * msec:0.00} ms");
    }
}
