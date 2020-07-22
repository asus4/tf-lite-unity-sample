using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace TensorFlowLite
{
    public class LandmarkDetect : BaseImagePredictor<float>
    {
        public struct Result
        {
            public float score;
            public Vector2[] joints;
        }

        public const int HAND_JOINT_NUM = 21;

        private float[] output0 = new float[HAND_JOINT_NUM * 2]; // keypoint
        private float[] output1 = new float[1]; // hand flag

        public LandmarkDetect(string modelPath) : base(modelPath, true)
        {

        }

        public override void Invoke(Texture inputTex)
        {
            //
            ToTensor(inputTex, input0);

            //
            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }

        public void Invoke(float[,,] input0)
        {

        }

        public Result GetResult()
        {
            var joints = new Vector2[HAND_JOINT_NUM];
            for (int i = 0; i < joints.Length; i++)
            {
                joints[i] = new Vector2(output0[i * 2], output0[i * 2 + 1]);
            }
            return new Result()
            {
                score = output1[0],
                joints = joints,
            };
        }
    }
}