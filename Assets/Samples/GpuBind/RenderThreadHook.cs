using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using AOT;
using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// Provides a simple way to execute C# code on Unity's render thread.
    /// </summary>
    public static class RenderThreadHook
    {
        static readonly IntPtr s_RenderThreadPtr = Marshal.GetFunctionPointerForDelegate<UnityRenderingEvent>(OnRenderThread);
        static readonly ConcurrentQueue<Action> s_Callbacks = new ConcurrentQueue<Action>();

        /// <summary>
        /// Default event ID called in the native plugin. Change this if conflicts with other systems.
        /// </summary>
        public static int HookEventId { get; set; } = 42;

        /// <summary>
        /// Execute an action on the render thread immediately using a specific event ID.
        /// Use different event IDs to avoid conflicts with other systems.
        /// </summary>
        /// <param name="callback">The action to execute on the render thread</param>
        public static void RunOnRenderThread(Action callback)
        {
            if (callback == null)
            {
                throw new ArgumentNullException(nameof(callback), "RenderThreadHook: Callback cannot be null");
            }

            // Enqueue callback for execution
            s_Callbacks.Enqueue(callback);
            GL.IssuePluginEvent(s_RenderThreadPtr, HookEventId);
        }

        /// <summary>
        /// Execute an action on the render thread asynchronously and wait for the frame to complete.
        /// This is useful when you need to ensure the render thread operation completes before continuing.
        /// Use with StartCoroutine() or inside another coroutine.
        /// </summary>
        /// <param name="callback">The action to execute on the render thread</param>
        /// <returns>Coroutine that waits for end of frame after executing the callback</returns>
        public static IEnumerator RunOnRenderThreadAsync(Action callback)
        {
            RunOnRenderThread(callback);
            yield return new WaitForEndOfFrame();
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void UnityRenderingEvent(int eventID);

        [MonoPInvokeCallback(typeof(UnityRenderingEvent))]
        private static void OnRenderThread(int eventID)
        {
            if (HookEventId != eventID)
            {
                return;
            }

            // Execute all queued callbacks
            while (s_Callbacks.TryDequeue(out Action callback))
            {
                callback?.Invoke();
            }
        }
    }
}
