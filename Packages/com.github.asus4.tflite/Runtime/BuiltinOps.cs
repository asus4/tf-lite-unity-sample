﻿/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

namespace TensorFlowLite
{
  /// <summary>
  /// tensorflow/lite/builtin_ops.h
  /// </summary>
  public enum BuiltinOperator
  {
    Add = 0,
    AveragePool2d = 1,
    Concatenation = 2,
    Conv2d = 3,
    DepthwiseConv2d = 4,
    DepthToSpace = 5,
    Dequantize = 6,
    EmbeddingLookup = 7,
    Floor = 8,
    FullyConnected = 9,
    HashtableLookup = 10,
    L2Normalization = 11,
    L2Pool2d = 12,
    LocalResponseNormalization = 13,
    Logistic = 14,
    LshProjection = 15,
    Lstm = 16,
    MaxPool2d = 17,
    Mul = 18,
    Relu = 19,
    ReluN1To1 = 20,
    Relu6 = 21,
    Reshape = 22,
    ResizeBilinear = 23,
    Rnn = 24,
    Softmax = 25,
    SpaceToDepth = 26,
    Svdf = 27,
    Tanh = 28,
    ConcatEmbeddings = 29,
    SkipGram = 30,
    Call = 31,
    Custom = 32,
    EmbeddingLookupSparse = 33,
    Pad = 34,
    UnidirectionalSequenceRnn = 35,
    Gather = 36,
    BatchToSpaceNd = 37,
    SpaceToBatchNd = 38,
    Transpose = 39,
    Mean = 40,
    Sub = 41,
    Div = 42,
    Squeeze = 43,
    UnidirectionalSequenceLstm = 44,
    StridedSlice = 45,
    BidirectionalSequenceRnn = 46,
    Exp = 47,
    TopkV2 = 48,
    Split = 49,
    LogSoftmax = 50,
    Delegate = 51,
    BidirectionalSequenceLstm = 52,
    Cast = 53,
    Prelu = 54,
    Maximum = 55,
    ArgMax = 56,
    Minimum = 57,
    Less = 58,
    Neg = 59,
    Padv2 = 60,
    Greater = 61,
    GreaterEqual = 62,
    LessEqual = 63,
    Select = 64,
    Slice = 65,
    Sin = 66,
    TransposeConv = 67,
    SparseToDense = 68,
    Tile = 69,
    ExpandDims = 70,
    Equal = 71,
    NotEqual = 72,
    Log = 73,
    Sum = 74,
    Sqrt = 75,
    Rsqrt = 76,
    Shape = 77,
    Pow = 78,
    ArgMin = 79,
    FakeQuant = 80,
    ReduceProd = 81,
    ReduceMax = 82,
    Pack = 83,
    LogicalOr = 84,
    OneHot = 85,
    LogicalAnd = 86,
    LogicalNot = 87,
    Unpack = 88,
    ReduceMin = 89,
    FloorDiv = 90,
    ReduceAny = 91,
    Square = 92,
    ZerosLike = 93,
    Fill = 94,
    FloorMod = 95,
    Range = 96,
    ResizeNearestNeighbor = 97,
    LeakyRelu = 98,
    SquaredDifference = 99,
    MirrorPad = 100,
    Abs = 101,
    SplitV = 102,
    Unique = 103,
    Ceil = 104,
    ReverseV2 = 105,
    AddN = 106,
    GatherNd = 107,
    Cos = 108,
    Where = 109,
    Rank = 110,
    Elu = 111,
    ReverseSequence = 112,
    MatrixDiag = 113,
    Quantize = 114,
    MatrixSetDiag = 115,
    Round = 116,
    HardSwish = 117,
    If = 118,
    While = 119,
    NonMaxSuppressionV4 = 120,
    NonMaxSuppressionV5 = 121,
    ScatterNd = 122,
    SelectV2 = 123,
    Densify = 124,
    SegmentSum = 125,
    BatchMatmul = 126,
    PlaceholderForGreaterOpCodes = 127,
    Cumsum = 128,
    CallOnce = 129,
    BroadcastTo = 130,
    Rfft2d = 131,
    Conv3d = 132,
    Imag = 133,
    Real = 134,
    ComplexAbs = 135,
    Hashtable = 136,
    HashtableFind = 137,
    HashtableImport = 138,
    HashtableSize = 139,
    ReduceAll = 140,
    Conv3dTranspose = 141,
    VarHandle = 142,
    ReadVariable = 143,
    AssignVariable = 144,
    BroadcastArgs = 145,
    RandomStandardNormal = 146,
    Bucketize = 147,
    RandomUniform = 148,
    Multinomial = 149,
    Gelu = 150,
    DynamicUpdateSlice = 151,
    Relu0To1 = 152,
    UnsortedSegmentProd = 153,
    UnsortedSegmentMax = 154,
    UnsortedSegmentSum = 155,
    Atan2 = 156,
    UnsortedSegmentMin = 157,
    Sign = 158,
    Bitcast = 159,
    BitwiseXor = 160,
    RightShift = 161,
    StablehloLogistic = 162,
    StablehloAdd = 163,
    StablehloDivide = 164,
    StablehloMultiply = 165,
    StablehloMaximum = 166,
    StablehloReshape = 167,
    StablehloClamp = 168,
    StablehloConcatenate = 169,
    StablehloBroadcastInDim = 170,
    StablehloConvolution = 171,
    StablehloSlice = 172,
    StablehloCustomCall = 173,
    StablehloReduce = 174,
    StablehloAbs = 175,
    StablehloAnd = 176,
    StablehloCosine = 177,
    StablehloExponential = 178,
    StablehloFloor = 179,
    StablehloLog = 180,
    StablehloMinimum = 181,
    StablehloNegate = 182,
    StablehloOr = 183,
    StablehloPower = 184,
    StablehloRemainder = 185,
    StablehloRsqrt = 186,
    StablehloSelect = 187,
    StablehloSubtract = 188,
    StablehloTanh = 189,
    StablehloScatter = 190,
    StablehloCompare = 191,
    StablehloConvert = 192,
    StablehloDynamicSlice = 193,
    StablehloDynamicUpdateSlice = 194,
    StablehloPad = 195,
    StablehloIota = 196,
    StablehloDotGeneral = 197,
    StablehloReduceWindow = 198,
    StablehloSort = 199,
    StablehloWhile = 200,
    StablehloGather = 201,
    StablehloTranspose = 202,
    Dilate = 203,
    StablehloRngBitGenerator = 204,
    ReduceWindow = 205,
    StablehloComposite = 206,
    StablehloShiftLeft = 207,
    StablehloCbrt = 208,
    StablehloCase = 209,
  }
}
