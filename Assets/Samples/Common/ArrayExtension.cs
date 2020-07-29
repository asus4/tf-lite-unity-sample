using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorFlowLite
{
    public static class ArrayExtension
    {
        public static Tuple<int, T>[] ToIndexValueTuple<T>(this T[] arr)
        {
            return arr.Select(o => Tuple.Create(0, o)).ToArray();
        }
    }
}
