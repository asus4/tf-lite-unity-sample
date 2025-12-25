using System;
using UnityEngine;
using UnityEngine.Rendering;
#if URP_10_OR_NEWER_ENABLED
using UnityEngine.Rendering.Universal;
#if URP_17_OR_NEWER_ENABLED
using UnityEngine.Rendering.RenderGraphModule;
#else
using UnityEngine.Experimental.Rendering;
#endif // URP_17_OR_NEWER_ENABLED
#else
using ScriptableRendererFeature = UnityEngine.ScriptableObject;
#endif // URP_10_OR_NEWER_ENABLED

namespace TensorFlowLite
{
    /// <summary>
    /// Ported from mock-arfoundation
    /// https://github.com/asus4/mock-arfoundation/blob/main/Packages/MockARFoundation/Runtime/ARBackgroundWebcamRendererFeature.cs
    /// </summary>
    public class CameraTextureBackgroundRendererFeature : ScriptableRendererFeature
    {
#if URP_10_OR_NEWER_ENABLED

        class PassData
        {
            // To restore original camera matrices after drawing the background
            internal Matrix4x4 worldToCameraMatrix;
            internal Matrix4x4 projectionMatrix;
            // To be rendered
            internal Mesh mesh;
            internal Material material;

        }

        /// <summary>
        /// The scriptable render pass to be added to the renderer when the camera background is to be rendered.
        /// </summary>
        CustomRenderPass m_ScriptablePass;

        /// <summary>
        /// The mesh for rendering the background shader.
        /// </summary>
        Mesh m_BackgroundMesh;



        /// <summary>
        /// Create the scriptable render pass.
        /// </summary>
        public override void Create()
        {
            m_ScriptablePass = new CustomRenderPass(RenderPassEvent.BeforeRenderingOpaques);

            m_BackgroundMesh = new Mesh();
            m_BackgroundMesh.vertices = new Vector3[]
            {
                new Vector3(0f, 0f, 0.1f),
                new Vector3(0f, 1f, 0.1f),
                new Vector3(1f, 1f, 0.1f),
                new Vector3(1f, 0f, 0.1f),
            };
            m_BackgroundMesh.uv = new Vector2[]
            {
                new Vector2(0f, 0f),
                new Vector2(0f, 1f),
                new Vector2(1f, 1f),
                new Vector2(1f, 0f),
            };
            m_BackgroundMesh.triangles = new int[] { 0, 1, 2, 0, 2, 3 };
        }

        /// <summary>
        /// Add the background rendering pass when rendering a game camera with an enabled AR camera background component.
        /// </summary>
        /// <param name="renderer">The scriptable renderer in which to enqueue the render pass.</param>
        /// <param name="renderingData">Additional rendering data about the current state of rendering.</param>
        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            Camera currentCamera = renderingData.cameraData.camera;
            if ((currentCamera != null) && (currentCamera.cameraType == CameraType.Game))
            {
                var cameraBackground = currentCamera.gameObject.GetComponent<CameraTextureBackground>();
                if ((cameraBackground != null) && (cameraBackground.Material != null))
                {
                    m_ScriptablePass.Setup(m_BackgroundMesh, cameraBackground.Material);
                    renderer.EnqueuePass(m_ScriptablePass);
                }
            }
        }

        /// <summary>
        /// The custom render pass to render the camera background.
        /// </summary>
        class CustomRenderPass : ScriptableRenderPass
        {
            const string CompatibilityScriptingAPIObsolete = "This rendering path is for compatibility mode only (when Render Graph is disabled). Use Render Graph API instead.";

            /// <summary>
            /// The name for the custom render pass which will be display in graphics debugging tools.
            /// </summary>
            const string k_CustomRenderPassName = "Camera Texture Background Pass (Render Graph Enabled)";

            const string k_CustomRenderPassNameObsolete = "Camera Texture Background Pass (Render Graph Disabled)";

            /// <summary>
            /// The orthogonal projection matrix for the background rendering.
            /// </summary>
            static readonly Matrix4x4 k_BackgroundOrthoProjection = Matrix4x4.Ortho(0f, 1f, 0f, 1f, -0.1f, 9.9f);

            PassData passData;


            /// <summary>
            /// Constructs the background render pass.
            /// </summary>
            /// <param name="renderPassEvent">The render pass event when this pass should be rendered.</param>
            public CustomRenderPass(RenderPassEvent renderPassEvent)
            {
                this.renderPassEvent = renderPassEvent;
            }

            /// <summary>
            /// Setup the background render pass.
            /// </summary>
            /// <param name="backgroundMesh">The mesh used for rendering the device background.</param>
            /// <param name="backgroundMaterial">The material used for rendering the device background.</param>
            /// <param name="invertCulling">Whether the culling mode should be inverted.</param>
            public void Setup(Mesh backgroundMesh, Material backgroundMaterial)
            {
                passData = new PassData()
                {
                    worldToCameraMatrix = Matrix4x4.identity,
                    projectionMatrix = Matrix4x4.identity,
                    mesh = backgroundMesh,
                    material = backgroundMaterial,
                };
            }

            /// <summary>
            /// Configure the render pass by configuring the render target and clear values.
            /// </summary>
            /// <param name="commandBuffer">The command buffer for configuration.</param>
            /// <param name="renderTextureDescriptor">The descriptor of the target render texture.</param>
            [Obsolete(CompatibilityScriptingAPIObsolete)]
            public override void Configure(CommandBuffer commandBuffer, RenderTextureDescriptor renderTextureDescriptor)
            {
                base.Configure(commandBuffer, renderTextureDescriptor);
                ConfigureClear(ClearFlag.Depth, Color.clear);
            }

            /// <summary>
            /// Execute the commands to render the camera background.
            /// </summary>
            /// <param name="context">The render context for executing the render commands.</param>
            /// <param name="renderingData">Additional rendering data about the current state of rendering.</param>
            [Obsolete(CompatibilityScriptingAPIObsolete)]
            public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
            {
                passData.worldToCameraMatrix = renderingData.cameraData.camera.worldToCameraMatrix;
                passData.projectionMatrix = renderingData.cameraData.camera.projectionMatrix;

                var cmd = CommandBufferPool.Get(k_CustomRenderPassNameObsolete);
                cmd.BeginSample(k_CustomRenderPassNameObsolete);

                ExecutePass(passData, CommandBufferHelpers.GetRasterCommandBuffer(cmd));

                cmd.EndSample(k_CustomRenderPassNameObsolete);
                context.ExecuteCommandBuffer(cmd);

                CommandBufferPool.Release(cmd);
            }

            static void ExecutePass(PassData data, RasterCommandBuffer cmd)
            {
                cmd.SetInvertCulling(false);
                cmd.SetViewProjectionMatrices(Matrix4x4.identity, k_BackgroundOrthoProjection);
                cmd.DrawMesh(data.mesh, Matrix4x4.identity, data.material);
                cmd.SetViewProjectionMatrices(data.worldToCameraMatrix, data.projectionMatrix);
            }

            /// <summary>
            /// Clean up any resources for the render pass.
            /// </summary>
            /// <param name="commandBuffer">The command buffer for frame cleanup.</param>
            public override void FrameCleanup(CommandBuffer commandBuffer)
            {
            }


#if URP_17_OR_NEWER_ENABLED
            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                var mesh = passData.mesh;
                var material = passData.material;

                using var builder = renderGraph.AddRasterRenderPass(k_CustomRenderPassName, out passData, profilingSampler);

                var cameraData = frameData.Get<UniversalCameraData>();
                var resourceData = frameData.Get<UniversalResourceData>();

                passData.worldToCameraMatrix = cameraData.camera.worldToCameraMatrix;
                passData.projectionMatrix = cameraData.camera.projectionMatrix;
                passData.mesh = mesh;
                passData.material = material;

                builder.AllowGlobalStateModification(true);
                builder.AllowPassCulling(false);

                builder.SetRenderAttachment(resourceData.activeColorTexture, 0);

                builder.SetRenderFunc(static (PassData data, RasterGraphContext context) => ExecutePass(data, context.cmd));
            }
#endif // URP_17_OR_NEWER_ENABLED
        }
#endif // URP_10_OR_NEWER_ENABLED
    }
}
