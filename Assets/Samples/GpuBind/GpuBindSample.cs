using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using TensorFlowLite;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class GpuBindSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "deeplabv3_257_mv_gpu.tflite";

    static readonly double msec = 1000.0 / Stopwatch.Frequency;

    IEnumerator Start()
    {
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

        var stopwatch = Stopwatch.StartNew();
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
            sb.AppendLine($"Prepere interpreter: {stopwatch.ElapsedTicks * msec:0.00} ms");
            stopwatch.Restart();

            var inputShape0 = interpreter.GetInputTensorInfo(0).shape;
            int height = inputShape0[1];
            int width = inputShape0[2];
            int channels = inputShape0[3];

            // interpreter.LogIOInfo();

            ComputeBuffer compute = null;
            if (useBinding)
            {
                compute = new ComputeBuffer(height * width * channels, sizeof(float));
                metalDelegate.BindBufferToTensor(0, compute);
                compute.Release();
            }
            else
            {
                var inputs = new float[height * width * channels];
                interpreter.SetInputTensorData(0, inputs);
            }
            sb.AppendLine($"Prepere data: {stopwatch.ElapsedTicks * msec:0.00} ms");
            stopwatch.Restart();

            interpreter.Invoke();

            sb.AppendLine($"Invoke {stopwatch.ElapsedTicks * msec:0.00} ms");

            compute?.Release();
            compute?.Dispose();
        }

        Debug.Log(sb.ToString());
    }
}
