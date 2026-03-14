// /src/engine/index.ts

// Core
export { Logger } from './core/Logger';
export type { LogLevel } from './core/Logger';

export { GPUBackend, FeatureTier } from './core/GPUBackend';
export type { GPUBackendConfig } from './core/GPUBackend';

export { defaultEngineConfig, mergeEngineConfig } from './core/EngineConfiguration';
export type { EngineConfiguration } from './core/EngineConfiguration';

export { ResourceManager } from './core/ResourceManager';
export type { ResourceHandle, BufferDescriptor, TextureDescriptor, SamplerDescriptor, LoadTextureOptions } from './core/ResourceManager';

// Shaders
export { ShaderSystem } from './shaders/ShaderSystem';
export type { ShaderHandle, ShaderDefines, ShaderSourceDescriptor, ShaderVariant, ShaderReflection, BindingInfo, VertexInputInfo } from './shaders/ShaderSystem';

// Pipelines
export { PipelineManager, buildVertexBufferLayouts, SEMANTIC_LOCATION,
         GBUFFER_COLOR_FORMATS, GBUFFER_DEPTH_FORMAT, SHADOW_DEPTH_FORMAT, HDR_COLOR_FORMAT,
         BLEND_PREMULTIPLIED_ALPHA, BLEND_STRAIGHT_ALPHA, BLEND_ADDITIVE } from './pipelines/PipelineManager';
export type { PipelineHandle, RenderPipelineKey, ComputePipelineKey, RayTracingPipelineKey } from './pipelines/PipelineManager';

// Geometry
export { MeshSystem, VertexSemantic } from './geometry/MeshSystem';
export { GeometryUtils } from './geometry/GeometryUtils';
export type { PlaneOptions, BoxOptions, UVSphereOptions, IcoSphereOptions } from './geometry/GeometryUtils';
export type { MeshHandle, MeshDescriptor, SubMesh, LODLevel, VertexLayoutDesc, VertexAttributeDesc, Vec3, AABB, BoundingSphere } from './geometry/MeshSystem';

// Materials
export { MaterialSystem, RenderPath, AlphaMode, MaterialShadingModel } from './materials/MaterialSystem';
export type { MaterialHandle, MaterialDescriptor, PBRParameters, MaterialTextures, MaterialRecord } from './materials/MaterialSystem';

// Scene
export { SceneGraph, NodeType } from './scene/SceneGraph';
export type { NodeHandle, SceneNode, Transform, CullResults, DrawableRef, Mat4, Quat, Vec3f } from './scene/SceneGraph';

// glTF Loader
export { GLTFLoader } from './scene/GLTFLoader';
export type { GLTFSkinData, GLTFAnimationData, GLTFChannelData } from './scene/GLTFLoader';

// Camera
export { CameraSystem, ProjectionType } from './camera/Camera';
export type { CameraHandle, CameraRecord, CameraDescriptor, PerspectiveParams, OrthographicParams, FrustumPlanes } from './camera/Camera';

// Lights
export { LightSystem, LightType, ShadowType } from './lights/LightSystem';
export type { LightHandle, LightDescriptor, LightRecord, ClusterBin } from './lights/LightSystem';

// Render Graph
export { RenderGraph, PassType, ResourceAccess, ResourceType } from './rendergraph/RenderGraph';
export type { PassHandle, VirtualResourceId, VirtualResource, RenderPass, PassExecuteFn, CompiledGraph,
              ColorAttachmentConfig, DepthAttachmentConfig } from './rendergraph/RenderGraph';

// Commands / Frame Orchestration
export { FrameOrchestrator } from './commands/FrameOrchestrator';
export type { FrameStats, PerFrameUniforms } from './commands/FrameOrchestrator';

// Post-Processing
export { PostProcessStack, BloomEffect, TonemapEffect, TAAEffect, FXAAEffect, SSREffect, SSAOEffect } from './postprocess/PostProcessStack';
export type { PostProcessEffect, PostProcessContext } from './postprocess/PostProcessStack';

// Environment / Background
export { BackgroundSystem, BackgroundType } from './environment/BackgroundSystem';
export type { BackgroundConfig } from './environment/BackgroundSystem';

// Animation
export { AnimationSystem } from './animation/AnimationSystem';
export type { AnimClipHandle, SkinHandle, PlaybackConfig, PlaybackState } from './animation/AnimationSystem';

// Scene Loader
export { SceneLoader } from './scene/SceneLoader';

// Input
export { InputSystem } from './input/InputSystem';
export type { GamepadState } from './input/InputSystem';
export { CameraControllerSystem } from './input/CameraController';
export type { InputDevice, CameraController } from './input/CameraController';

// Engine (top-level)
export { Engine } from './Engine';
export type { EngineConfig, UpdateContext, UpdateCallback } from './Engine';