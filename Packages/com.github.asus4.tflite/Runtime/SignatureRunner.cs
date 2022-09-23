/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TfLiteInterpreter = System.IntPtr;
using TfLiteTensor = System.IntPtr;
using TfLiteSignatureRunner = System.IntPtr;
using UnityEngine.Assertions;

namespace TensorFlowLite
{
    /// <summary>
    /// C# bindings for SignatureRunner APIs
    /// https://www.tensorflow.org/lite/guide/signatures
    /// </summary>
    public class SignatureRunner : Interpreter
    {
        private TfLiteSignatureRunner runner;

        // Mappings of signature_name -> tensor_index
        private Dictionary<string, int> inputTensors;
        private Dictionary<string, int> outputTensors;

        public string[] InputSignatureNames { get; private set; }
        public string[] OutputSignatureNames { get; private set; }

        public SignatureRunner(string signatureName, byte[] modelData, InterpreterOptions options)
            : base(modelData, options)
        {
            Initialize(signatureName);
        }

        public SignatureRunner(int signatureIndex, byte[] modelData, InterpreterOptions options)
            : base(modelData, options)
        {
            Initialize(GetSignatureKey(signatureIndex));
        }

        public override void Dispose()
        {
            if (runner != TfLiteSignatureRunner.Zero)
            {
                TfLiteSignatureRunnerDelete(runner);
            }
            inputTensors?.Clear();
            outputTensors?.Clear();
            base.Dispose();
        }

        public int GetSignatureCount()
        {
            return TfLiteInterpreterGetSignatureCount(InterpreterPointer);
        }

        public string GetSignatureKey(int index)
        {
            return ToString(TfLiteInterpreterGetSignatureKey(InterpreterPointer, index));
        }

        public int GetSignatureInputCount()
        {
            return TfLiteSignatureRunnerGetInputCount(runner);
        }

        public string GetSignatureInputName(int index)
        {
            return ToString(TfLiteSignatureRunnerGetInputName(runner, index));
        }

        public void SetSignatureInputTensorData(string name, Array inputTensorData)
        {
            if (!inputTensors.TryGetValue(name, out int tensorIndex))
            {
                throw new ArgumentException($"{name} is not a valid input tensor name");
            }
            SetInputTensorData(tensorIndex, inputTensorData);
        }

        public void ResizeSignatureInputTensor(string inputName, int[] inputDims)
        {
            ThrowIfError(TfLiteSignatureRunnerResizeInputTensor(runner, inputName, inputDims, inputDims.Length));
        }

        public void AllocateSignatureTensors()
        {
            ThrowIfError(TfLiteSignatureRunnerAllocateTensors(runner));
        }

        private TfLiteTensor GetSignatureInputTensor(string inputName)
        {
            return TfLiteSignatureRunnerGetInputTensor(runner, inputName);
        }

        public TensorInfo GetSignatureInputInfo(string inputName)
        {
            TfLiteTensor tensor = GetSignatureInputTensor(inputName);
            return GetTensorInfo(tensor);
        }

        public override void Invoke()
        {
            ThrowIfError(TfLiteSignatureRunnerInvoke(runner));
        }

        public int GetSignatureOutputCount()
        {
            return TfLiteSignatureRunnerGetOutputCount(runner);
        }

        public string GetSignatureOutputName(int index)
        {
            return ToString(TfLiteSignatureRunnerGetOutputName(runner, index));
        }

        public void GetSignatureOutputTensorData(string name, Array outputTensorData)
        {
            if (!outputTensors.TryGetValue(name, out int tensorIndex))
            {
                throw new ArgumentException($"{name} is not a valid output tensor name");
            }
            GetOutputTensorData(tensorIndex, outputTensorData);
        }

        private TfLiteTensor GetSignatureOutputTensor(string outputName)
        {
            return TfLiteSignatureRunnerGetOutputTensor(runner, outputName);
        }

        public TensorInfo GetSignatureOutputInfo(string outputName)
        {
            TfLiteTensor tensor = GetSignatureOutputTensor(outputName);
            return GetTensorInfo(tensor);
        }

        private static string ToString(IntPtr ptr)
        {
            return Marshal.PtrToStringAnsi(ptr);
        }

        private void Initialize(string signatureName)
        {
            runner = TfLiteInterpreterGetSignatureRunner(InterpreterPointer, signatureName);
            if (runner == TfLiteSignatureRunner.Zero)
            {
                throw new Exception("Failed to create SignatureRunner");
            }

            inputTensors = CreateMap(isInput: true);
            outputTensors = CreateMap(isInput: false);

            InputSignatureNames = inputTensors.Keys.ToArray();
            OutputSignatureNames = outputTensors.Keys.ToArray();
        }

        private Dictionary<string, int> CreateMap(bool isInput)
        {
            int signatureCount = isInput ? GetSignatureInputCount() : GetSignatureOutputCount();
            int tensorCount = isInput ? GetInputTensorCount() : GetOutputTensorCount();
            Assert.AreEqual(signatureCount, tensorCount);

            var indexMap = new Dictionary<TfLiteTensor, int>(tensorCount);
            for (int i = 0; i < tensorCount; i++)
            {
                TfLiteTensor tensor = isInput ? GetInputTensor(i) : GetOutputTensor(i);
                indexMap.Add(tensor, i);
            }

            var map = new Dictionary<string, int>(signatureCount);
            for (int i = 0; i < signatureCount; i++)
            {
                string name = isInput ? GetSignatureInputName(i) : GetSignatureOutputName(i);
                TfLiteTensor tensor = isInput ? GetSignatureInputTensor(name) : GetSignatureOutputTensor(name);

                if (indexMap.TryGetValue(tensor, out int index))
                {
                    map.Add(name, index);
                }
                else
                {
                    throw new Exception($"Failed to find tensor for signature {name}");
                }
            }

            return map;
        }

        /// --------------------------------------------------------------------------
        /// SignatureRunner APIs
        ///
        /// You can run inference by either:
        ///
        /// (i) (recommended) using the Interpreter to initialize SignatureRunner(s) and
        ///     then only using SignatureRunner APIs.
        ///
        /// (ii) only using Interpreter APIs.
        ///
        /// NOTE:
        /// * Only use one of the above options to run inference, i.e, avoid mixing both
        ///   SignatureRunner APIs and Interpreter APIs to run inference as they share
        ///   the same underlying data (e.g. updating an input tensor “A” retrieved
        ///   using the Interpreter APIs will update the state of the input tensor “B”
        ///   retrieved using SignatureRunner APIs, if they point to the same underlying
        ///   tensor in the model; as it is not possible for a user to debug this by
        ///   analyzing the code, it can lead to undesirable behavior).
        /// * The TfLiteSignatureRunner type is conditionally thread-safe, provided that
        ///   no two threads attempt to simultaneously access two TfLiteSignatureRunner
        ///   instances that point to the same underlying signature, or access a
        ///   TfLiteSignatureRunner and its underlying TfLiteInterpreter, unless all
        ///   such simultaneous accesses are reads (rather than writes).
        /// * The lifetime of a TfLiteSignatureRunner object ends when
        ///   TfLiteSignatureRunnerDelete() is called on it (or when the lifetime of the
        ///   underlying TfLiteInterpreter ends -- but you should call
        ///   TfLiteSignatureRunnerDelete() before that happens in order to avoid
        ///   resource leaks).
        /// * You can only apply delegates to the interpreter (via
        ///   TfLiteInterpreterOptions) and not to a signature.

        [DllImport(TensorFlowLibrary)]
        private static extern int TfLiteInterpreterGetSignatureCount(TfLiteInterpreter interpreter);

        [DllImport(TensorFlowLibrary)]
        private static extern IntPtr TfLiteInterpreterGetSignatureKey(TfLiteInterpreter interpreter, int signature_index);

        [DllImport(TensorFlowLibrary)]
        private static extern TfLiteSignatureRunner TfLiteInterpreterGetSignatureRunner(TfLiteInterpreter interpreter, string signature_name);

        [DllImport(TensorFlowLibrary)]
        private static extern int TfLiteSignatureRunnerGetInputCount(TfLiteSignatureRunner signature_runner);

        [DllImport(TensorFlowLibrary)]
        private static extern IntPtr TfLiteSignatureRunnerGetInputName(TfLiteSignatureRunner signature_runner, int input_index);

        [DllImport(TensorFlowLibrary)]
        private static extern Status TfLiteSignatureRunnerResizeInputTensor(
            TfLiteSignatureRunner signatureRunner, string input_name,
            int[] input_dims, int input_dims_size);

        [DllImport(TensorFlowLibrary)]
        private static extern Status TfLiteSignatureRunnerAllocateTensors(TfLiteSignatureRunner signature_runner);

        [DllImport(TensorFlowLibrary)]
        private static extern TfLiteTensor TfLiteSignatureRunnerGetInputTensor(TfLiteSignatureRunner signature_runner, string input_name);

        [DllImport(TensorFlowLibrary)]
        private static extern Status TfLiteSignatureRunnerInvoke(TfLiteSignatureRunner signature_runner);

        [DllImport(TensorFlowLibrary)]
        private static extern int TfLiteSignatureRunnerGetOutputCount(TfLiteSignatureRunner signature_runner);

        [DllImport(TensorFlowLibrary)]
        private static extern IntPtr TfLiteSignatureRunnerGetOutputName(TfLiteSignatureRunner signature_runner, int output_index);

        [DllImport(TensorFlowLibrary)]
        private static extern TfLiteTensor TfLiteSignatureRunnerGetOutputTensor(TfLiteSignatureRunner signature_runner, string output_name);

        [DllImport(TensorFlowLibrary)]
        private static extern void TfLiteSignatureRunnerDelete(TfLiteSignatureRunner signature_runner);
    }
}
