using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorFlowLite
{
    public static class ArrayExtension
    {
        public static IEnumerable<Tuple<int, T>> ToIndexValueTuple<T>(this T[] arr)
        {
            int index = 0;
            return arr.Select(o =>
            {
                return Tuple.Create(index++, o);
            });
        }
    }
}
