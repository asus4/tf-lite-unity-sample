namespace TensorFlowLite.MoveNet
{
    using UnityEngine;

    public class MoveNetPose
    {
        public enum Part
        {
            NOSE = 0,
            LEFT_EYE = 1,
            RIGHT_EYE = 2,
            LEFT_EAR = 3,
            RIGHT_EAR = 4,
            LEFT_SHOULDER = 5,
            RIGHT_SHOULDER = 6,
            LEFT_ELBOW = 7,
            RIGHT_ELBOW = 8,
            LEFT_WRIST = 9,
            RIGHT_WRIST = 10,
            LEFT_HIP = 11,
            RIGHT_HIP = 12,
            LEFT_KNEE = 13,
            RIGHT_KNEE = 14,
            LEFT_ANKLE = 15,
            RIGHT_ANKLE = 16,
        }

        public static readonly Part[,] Connections = new Part[,]
        {
            // HEAD
            { Part.LEFT_EAR, Part.LEFT_EYE },
            { Part.LEFT_EYE, Part.NOSE },
            { Part.NOSE, Part.RIGHT_EYE },
            { Part.RIGHT_EYE, Part.RIGHT_EAR },
            // BODY
            { Part.LEFT_HIP, Part.LEFT_SHOULDER },
            { Part.LEFT_ELBOW, Part.LEFT_SHOULDER },
            { Part.LEFT_ELBOW, Part.LEFT_WRIST },
            { Part.LEFT_HIP, Part.LEFT_KNEE },
            { Part.LEFT_KNEE, Part.LEFT_ANKLE },
            { Part.RIGHT_HIP, Part.RIGHT_SHOULDER },
            { Part.RIGHT_ELBOW, Part.RIGHT_SHOULDER },
            { Part.RIGHT_ELBOW, Part.RIGHT_WRIST },
            { Part.RIGHT_HIP, Part.RIGHT_KNEE },
            { Part.RIGHT_KNEE, Part.RIGHT_ANKLE },
            { Part.LEFT_SHOULDER, Part.RIGHT_SHOULDER },
            { Part.LEFT_HIP, Part.RIGHT_HIP }
        };

        public readonly struct Joint
        {
            public readonly float x;
            public readonly float y;
            public readonly float score;

            public Joint(float x, float y, float score)
            {
                this.x = x;
                this.y = y;
                this.score = score;
            }
        }

        public const int JOINT_COUNT = 17;

        // x, y, confidence
        public readonly Joint[] joints = new Joint[JOINT_COUNT];

        public int Length => joints.Length;

        public Joint this[int index]
        {
            get => joints[index];
            set => joints[index] = value;
        }
    }

    public class MoveNetPoseWithBoundingBox : MoveNetPose
    {
        // Normalized coordinates
        public Rect boundingBox;
        public float score;
    }
}
