// Currently supported only on Android or macOS Editor
#if UNITY_ANDROID || UNITY_EDITOR_OSX

using System;
using System.Collections;
using System.Runtime.InteropServices;
using AOT;
using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// Provides a simple way to execute C# code on Unity's render thread.
    /// Call RunOnRenderThread with an Action to execute it immediately on the render thread.
    /// 
    /// See plugin source here:
    /// https://github.com/asus4/RenderThreadHookPlugin
    /// </summary>
    public static class RenderThreadHook
    {
        static Action s_ManagedCallback = null;

        /// <summary>
        /// Default event ID called in the native plugin. Change this if conflicts with other systems.
        /// </summary>
        public static int HookEventId { get; set; } = 42;

        /// <summary>
        /// Execute an action on the render thread immediately using a specific event ID.
        /// The callback is registered and executed in a single call.
        /// Use different event IDs to avoid conflicts with other systems.
        /// </summary>
        /// <param name="callback">The action to execute on the render thread</param>
        public static void RunOnRenderThread(Action callback)
        {
            if (callback == null)
            {
                throw new ArgumentNullException(nameof(callback), "RenderThreadHook: Callback cannot be null");
            }
            int eventId = HookEventId;

            try
            {
                // Register and execute immediately
                s_ManagedCallback = callback;
                RegisterCallback(eventId, StaticCallbackHandler);
                GL.IssuePluginEvent(GetRenderEventFunc(), eventId);
            }
            catch (Exception e)
            {
                Debug.LogError($"RenderThreadHook: Failed to execute on render thread - {e.Message}");
                s_ManagedCallback = null;
            }
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

        /// <summary>
        /// Unregister the current callback. This is typically not needed as callbacks
        /// are executed immediately, but can be used for cleanup.
        /// </summary>
        public static void Unregister()
        {
            try
            {
                UnregisterCallback();
                s_ManagedCallback = null;
            }
            catch (Exception e)
            {
                Debug.LogError($"RenderThreadHook: Failed to unregister callback - {e.Message}");
            }
        }

        // Static callback handler that will be called from C++
        [MonoPInvokeCallback(typeof(Action))]
        static void StaticCallbackHandler()
        {
            if (s_ManagedCallback == null)
            {
                throw new InvalidOperationException("RenderThreadHook: No callback registered");
            }
            s_ManagedCallback.Invoke();
        }

        // Plugin name - different for each platform
#if UNITY_ANDROID && !UNITY_EDITOR
        const string PluginName = "RenderThreadHook";
#elif UNITY_IOS && !UNITY_EDITOR
        const string PluginName = "__Internal";
#else
        const string PluginName = "RenderThreadHook";
#endif

        // Native plugin function imports
        [DllImport(PluginName)]
        static extern void RegisterCallback(int eventId, Action callback);

        [DllImport(PluginName)]
        static extern void UnregisterCallback();

        [DllImport(PluginName)]
        static extern IntPtr GetRenderEventFunc();
    }
}

#endif // UNITY_ANDROID || UNITY_EDITOR_OSX
