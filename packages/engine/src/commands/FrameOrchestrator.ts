// /src/engine/commands/FrameOrchestrator.ts

import gbufferWGSL              from '../shaders/gbuffer.wgsl?raw';
import depthPrepassWGSL         from '../shaders/depth_prepass.wgsl?raw';
import deferredLightingWGSL     from '../shaders/deferred_lighting.wgsl?raw';
import shadowWGSL               from '../shaders/shadow.wgsl?raw';
import blitWGSL                 from '../shaders/blit.wgsl?raw';
import depthDownsampleWGSL      from '../shaders/depth_downsample.wgsl?raw';
import forwardTransparentWGSL   from '../shaders/forward_transparent.wgsl?raw';
import shadowTransparentWGSL    from '../shaders/shadow_transparent.wgsl?raw';
import fullscreenCopyWGSL       from '../shaders/fullscreen_copy.wgsl?raw';
import type { BackgroundSystem } from '../environment/BackgroundSystem';
import { Logger } from '../core/Logger';
import type { InputSystem } from '../input/InputSystem';
import type { CameraControllerSystem } from '../input/CameraController';
import { VertexSemantic } from '../geometry/MeshSystem';
import type { GPUBackend } from '../core/GPUBackend';
import type { ResourceManager, ResourceHandle } from '../core/ResourceManager';
import type { ShaderSystem, ShaderDefines } from '../shaders/ShaderSystem';
import { BLEND_STRAIGHT_ALPHA, buildVertexBufferLayouts } from '../pipelines/PipelineManager';
import type { PipelineManager } from '../pipelines/PipelineManager';
import type { MeshSystem, VertexLayoutDesc, MeshHandle } from '../geometry/MeshSystem';
import { PassType, type RenderGraph } from '../rendergraph/RenderGraph';
import type { SceneGraph, CullResults, DrawableRef } from '../scene/SceneGraph';
import { RenderPath } from '../materials/MaterialSystem';
import type { MaterialHandle } from '../materials/MaterialSystem';
import type { CameraSystem } from '../camera/Camera';
import type { LightSystem } from '../lights/LightSystem';
import type { MaterialSystem } from '../materials/MaterialSystem';
import type { PostProcessStack, PostProcessContext } from '../postprocess/PostProcessStack';
import type { EngineConfiguration } from '../core/EngineConfiguration';
import type { AnimationSystem } from '../animation/AnimationSystem';
import type { Engine } from '../Engine';
import { BindGroupCache } from '../core/BindGroupCache';

// -------------------------------------------------------------------------
// Per-frame uniform buffer layout (must match the WGSL PerFrameUniforms struct)
// -------------------------------------------------------------------------
//
//   viewMatrix            : mat4x4f  = 64 bytes  @ offset   0
//   projectionMatrix      : mat4x4f  = 64 bytes  @ offset  64
//   viewProjectionMatrix  : mat4x4f  = 64 bytes  @ offset 128
//   inverseViewProjection : mat4x4f  = 64 bytes  @ offset 192
//   cameraPosition        : vec3f    = 12 bytes  @ offset 256  (+ 4 pad)
//   time                  : f32      =  4 bytes  @ offset 272
//   deltaTime             : f32      =  4 bytes  @ offset 276
//   resolution            : vec2f    =  8 bytes  @ offset 280
//   frameIndex            : f32      =  4 bytes  @ offset 288
//   exposure              : f32      =  4 bytes  @ offset 292
//   jitter                : vec2f    =  8 bytes  @ offset 296
//                                      ——————————
//   Total                            = 304 bytes (19 × 16, 16-byte aligned)
//
const PER_FRAME_BUFFER_SIZE = 304; // bytes

// -------------------------------------------------------------------------
// Model matrix buffer constants
// -------------------------------------------------------------------------
//
// Each draw slot is padded to 256 bytes (minUniformBufferOffsetAlignment).
// The G-Buffer pipeline reads only the first 64 bytes (mat4x4f) per slot.
//
const MAX_DRAWS            = 4096;
const MODEL_UNIFORM_STRIDE = 256; // bytes per slot
const MODEL_MATRIX_SIZE    = 64;  // bytes — mat4x4f

// Maximum number of static instances that can be batched per frame.
// Each slot is 64 bytes (mat4x4f); total = 256 KB.
const MAX_STATIC_INSTANCES = 4096;

// Shadow VP buffer: one 64-byte slot (mat4x4f) per cascade/face.
// 32 slots covers e.g. 5 cube-map lights (30 faces) or mixed light types.
const SHADOW_VP_BUFFER_SIZE = 64; // bytes — mat4x4f
const MAX_SHADOW_CASCADE_SLOTS = 32;

// -------------------------------------------------------------------------
// Internal types
// -------------------------------------------------------------------------

/**
 * One GPU draw call's worth of static (instanced) geometry.
 * All instances share the same mesh and material; their world matrices are
 * packed consecutively in the static instance storage buffer starting at
 * `firstInstance`.
 */
interface InstanceBatch {
    meshHandle:     MeshHandle;
    materialHandle: MaterialHandle;
    /** Base material shader defines — USE_INSTANCING is added on top. */
    baseDefines:    ShaderDefines;
    firstInstance:  number;
    instanceCount:  number;
}

// Standard interleaved vertex layout expected by gbuffer.wgsl
// (stride 48: position float32x3 | normal float32x3 | tangent float32x4 | uv0 float32x2)
const STANDARD_VERTEX_LAYOUT: VertexLayoutDesc[] = [{
    arrayStride: 48,
    attributes: [
        { semantic: VertexSemantic.Position, format: 'float32x3', offset:  0 },
        { semantic: VertexSemantic.Normal,   format: 'float32x3', offset: 12 },
        { semantic: VertexSemantic.Tangent,  format: 'float32x4', offset: 24 },
        { semantic: VertexSemantic.UV0,      format: 'float32x2', offset: 40 },
    ],
}];

// Skinned vertex layout: standard layout + skinning data in a second vertex buffer
// (stride 24: joints uint16x4 + weights float32x4)
const SKINNED_VERTEX_LAYOUT: VertexLayoutDesc[] = [
    STANDARD_VERTEX_LAYOUT[0]!,
    {
        arrayStride: 24,
        attributes: [
            { semantic: VertexSemantic.Joints0,  format: 'uint16x4',  offset: 0 },
            { semantic: VertexSemantic.Weights0, format: 'float32x4', offset: 8 },
        ],
    },
];

// -------------------------------------------------------------------------
// Types
// -------------------------------------------------------------------------

export interface FrameStats {
    frameIndex: number;
    gpuTimeMs:  number;
    cpuTimeMs:  number;
    /** Actual frames per second measured from real frame-to-frame intervals. */
    realFps:    number;
    drawCalls:  number;
    triangles:  number;
    passCount:  number;
}

export interface PerFrameUniforms {
    viewMatrix:            Float32Array;
    projectionMatrix:      Float32Array;
    viewProjectionMatrix:  Float32Array;
    inverseViewProjection: Float32Array;
    cameraPosition:        Float32Array;
    time:                  number;
    deltaTime:             number;
    resolution:            [number, number];
    frameIndex:            number;
    exposure:              number;
    jitter:                [number, number];
}

// -------------------------------------------------------------------------
// FrameOrchestrator
// -------------------------------------------------------------------------

/**
 * Orchestrates the high-level frame loop:
 *
 *  1. Begin frame  → advance frame index, reset render graph
 *  2. Update scene → propagate transforms, cull, update camera
 *  3. Build graph  → register all passes into the RenderGraph
 *  4. Compile      → topological sort, dead-pass culling, resource allocation
 *  5. Execute      → encode command buffers, submit
 *  6. End frame    → update stats
 */
export class FrameOrchestrator {

    private readonly _log = new Logger('FrameOrchestrator');

    // --- dependencies (injected) -------------------------------------------
    private _backend!:          GPUBackend;
    private _resources!:        ResourceManager;
    private _shaderSystem!:     ShaderSystem;
    private _pipelineManager!:  PipelineManager;
    private _meshSystem!:       MeshSystem;
    private _renderGraph!:      RenderGraph;
    private _sceneGraph!:       SceneGraph;
    private _cameraSystem!:     CameraSystem;
    private _lightSystem!:      LightSystem;
    private _materialSystem!:   MaterialSystem;
    private _postProcessStack!: PostProcessStack;
    private _engine:             Engine | null = null;
    // --- frame state --------------------------------------------------------
    private _frameIndex:     number  = 0;
    private _lastTimestamp:  number  = 0;
    private _running:        boolean = false;
    private _rafId:          number  = 0;

    // --- per-frame uniform ring (one buffer + bind group per frame-in-flight)
    private _perFrameBuffers:          GPUBuffer[]        = [];
    private _perFrameBindGroups:       GPUBindGroup[]     = [];
    private _perFrameBindGroupLayout!: GPUBindGroupLayout;

    // --- G-Buffer pipeline + per-draw resources ----------------------------
    /**
     * Pipeline variant cache keyed by a sorted define string, e.g.
     * "HAS_BASE_COLOR_MAP=1|HAS_TEXTURES=1".  Each variant is compiled once
     * and reused.  The no-define variant (key "") is always pre-warmed in init().
     */
    private _gbufferPipelineCache: Map<string, GPURenderPipeline> = new Map();
    /**
     * Per-variant per-frame bind groups for @group(0).
     * Key = variant cache key.  Value = GPUBindGroup[] of length maxFramesInFlight.
     * Auto-layout BGLs are pipeline-specific, so each variant needs its own set.
     */
    private _gbufferPerFrameBGCache: Map<string, GPUBindGroup[]> = new Map();
    /**
     * Per-variant model-matrix bind groups for @group(2).
     * Key = variant cache key.  Value = GPUBindGroup[] of length MAX_DRAWS.
     */
    private _gbufferModelMatBGCache: Map<string, GPUBindGroup[]> = new Map();
    /** Single buffer holding MAX_DRAWS model matrices (stride 256 bytes). */
    private _modelMatBuffer:     GPUBuffer | null = null;

    // --- Static instancing resources ----------------------------------------
    /**
     * GPU storage buffer holding world matrices for all static (instanced)
     * drawables visible this frame.  Shader reads instanceMatrices[instance_index].
     */
    private _staticInstanceBuffer: GPUBuffer | null = null;
    /**
     * Per-GBuffer-variant single bind group for @group(2) in the USE_INSTANCING
     * pipeline — points to the full _staticInstanceBuffer (no sub-range).
     * Key = variant cache key (with USE_INSTANCING included).
     */
    private _instancedG2BGCache: Map<string, GPUBindGroup> = new Map();
    /** USE_INSTANCING variant of the depth prepass pipeline. */
    private _depthPrepassInstancedPipeline: GPURenderPipeline | null = null;
    /** Per-frame bind groups for @group(0) in the instanced depth prepass pipeline. */
    private _depthPrepassInstancedPerFrameBGs: GPUBindGroup[] = [];
    /** @group(2) bind group for the instanced depth prepass — full storage buffer. */
    private _depthPrepassInstancedG2BG: GPUBindGroup | null = null;

    // --- Per-frame instance batching results --------------------------------
    /** Packed instanced draw calls built each frame from static opaqueDrawables. */
    private _instanceBatches:  InstanceBatch[] = [];
    /** Non-static drawables that use the per-draw uniform path. */
    private _dynamicDrawables: DrawableRef[]   = [];

    // --- Deferred lighting pipeline ----------------------------------------
    private _lightingPipeline:           GPURenderPipeline | null = null;
    /** Per-frame bind groups for @group(0) in the lighting pipeline. */
    private _lightingPerFrameBindGroups: GPUBindGroup[] = [];
    /** BGL for @group(1) (G-Buffer textures) derived from the lighting pipeline. */
    private _lightingGBufferBGL:         GPUBindGroupLayout | null = null;
    /** BGL for @group(2) (light storage buffer) derived from the lighting pipeline. */
    private _lightingLightsBGL:          GPUBindGroupLayout | null = null;
    /** Persistent bind group wrapping the light storage buffer. */
    private _lightingLightsBG:           GPUBindGroup | null = null;
    /** Nearest-clamp sampler shared by the lighting pass G-Buffer reads. */
    private _clampSampler:               GPUSampler | null = null;

    // --- Shadow map pipelines + per-cascade resources ----------------------
    /** Depth-only shadow pipeline (non-instanced). */
    private _shadowPipeline:          GPURenderPipeline | null = null;
    /** Depth-only shadow pipeline (USE_INSTANCING=1 — reads storage buffer). */
    private _shadowInstancedPipeline: GPURenderPipeline | null = null;
    /** Depth-only shadow pipeline for skinned meshes (USE_SKINNING=1). */
    private _shadowSkinnedPipeline:   GPURenderPipeline | null = null;
    /** @group(0) bind groups for the skinned shadow pipeline, one per cascade slot. */
    private _shadowVPSkinnedBGs:      GPUBindGroup[]     = [];
    /** Shadow pipeline cache for transparent materials (has fragment shader with discard). */
    private _shadowTransparentPipelineCache: Map<string, GPURenderPipeline> = new Map();
    /** @group(0) bind groups for the transparent shadow pipeline, one per cascade slot. */
    private _shadowVPTransparentBGs:  GPUBindGroup[]     = [];
    /** @group(2) model matrix bind groups for transparent shadow draws. */
    private _shadowTransparentModelMatBGs: GPUBindGroup[] = [];
    /**
     * Per-cascade 64-byte uniform buffers holding the light VP matrix.
     * Index: cascade slot (0…MAX_SHADOW_CASCADE_SLOTS-1).
     */
    private _shadowVPBuffers:         GPUBuffer[]        = [];
    /** @group(0) bind groups for the non-instanced shadow pipeline, one per cascade slot. */
    private _shadowVPBGs:             GPUBindGroup[]     = [];
    /** @group(0) bind groups for the instanced shadow pipeline, one per cascade slot. */
    private _shadowVPInstBGs:         GPUBindGroup[]     = [];
    /** @group(2) bind groups for the non-instanced shadow pipeline, one per draw slot. */
    private _shadowModelMatBGs:       GPUBindGroup[]     = [];
    /** @group(2) bind group for the instanced shadow pipeline — full static instance buffer. */
    private _shadowInstancedG2BG:     GPUBindGroup | null = null;
    /** Comparison sampler for PCF shadow reads in the lighting pass. */
    private _shadowCompSampler:       GPUSampler | null  = null;

    // --- Blit pipeline (HDR → swap-chain with tonemapping) -----------------
    private _blitPipeline: GPURenderPipeline | null = null;
    /** BGL for @group(0) (colour texture + sampler + exposure uniform) in the blit pipeline. */
    private _blitBGL:      GPUBindGroupLayout | null = null;
    /** Tiny uniform buffer holding a single f32 exposure value for the blit shader. */
    private _blitExposureBuffer: GPUBuffer | null = null;

    // --- depth downsample compute pipeline -----------------------------------
    private _depthDownsamplePipeline: GPUComputePipeline | null = null;

    // --- depth prepass pipeline + per-frame / per-draw bind groups ----------
    // The prepass uses @group(0) (per-frame) and @group(2) (model matrix) with
    // auto-derived BGLs that differ from the G-Buffer pipeline's BGLs, so
    // separate bind groups are required (same underlying GPU buffers).
    private _depthPrepassPipeline:          GPURenderPipeline | null = null;
    private _depthPrepassPerFrameBindGroups: GPUBindGroup[] = [];
    private _depthPrepassModelMatBindGroups: GPUBindGroup[] = [];

    // --- input & camera controllers (optional) -----------------------------
    private _inputSystem:        InputSystem | null = null;
    private _cameraControllers:  CameraControllerSystem | null = null;

    // --- animation system (injected) ------------------------------------------
    private _animationSystem:    AnimationSystem | null = null;

    // --- background system (injected) ----------------------------------------
    private _backgroundSystem:   BackgroundSystem | null = null;
    /** 1×1 white texture view used as placeholder when no bg texture is set. */
    private _placeholder2DView:  GPUTextureView | null = null;

    /** Cached inverse VP for background cubemap pass (updated each frame). */
    private _lastInvViewProj: Float32Array = new Float32Array(16);

    // --- environment cubemap for IBL reflections ----------------------------
    /** BGL for @group(3) (env cubemap) derived from the lighting pipeline. */
    private _lightingEnvBGL:    GPUBindGroupLayout | null = null;
    /** @group(3) bind group: envParams uniform + cubemap + sampler. */
    private _lightingEnvBG:     GPUBindGroup | null = null;
    /** Uniform buffer for EnvParams (16 bytes: enabled u32 + ambient rgb f32×3). */
    private _envParamsBuffer:   GPUBuffer | null = null;
    /** Current ambient color (linear RGB). Stored so we can re-upload when cubemap changes. */
    private _ambientColor: [number, number, number] = [0.03, 0.03, 0.03];
    /** 1×1 placeholder cube texture used when no env cubemap is set. */
    private _placeholderCubeTex: GPUTexture | null = null;
    /** Linear sampler for env cubemap. */
    private _envCubeSampler:    GPUSampler | null = null;

    // --- forward transparent pass resources -----------------------------------
    /** Fullscreen copy pipeline (HDR → HDR, no tonemapping). */
    private _fullscreenCopyPipeline: GPURenderPipeline | null = null;
    /** Forward transparent pipeline cache: variantKey → pipeline. */
    private _forwardPipelineCache: Map<string, GPURenderPipeline> = new Map();
    /** @group(2) model matrix bind groups for forward (one per draw slot). */
    private _forwardModelMatBGs: GPUBindGroup[] = [];
    /** @group(3) bind group layout for forward transparent pass. */
    private _forwardLightEnvBGL: GPUBindGroupLayout | null = null;
    /** @group(3) bind group: lights + shadows + env (combined). */
    private _forwardLightEnvBG: GPUBindGroup | null = null;

    // --- engine configuration (injected) -------------------------------------
    private _config: EngineConfiguration | null = null;

    // --- cull results for this frame (used by render callbacks) ------------
    private _cullResults: CullResults = {
        opaqueDrawables:      [],
        transparentDrawables: [],
        visibleLights:        [],
    };

    // --- configurable limits (from EngineConfiguration, applied in init) ------
    private _maxDraws:           number = MAX_DRAWS;
    private _maxStaticInstances: number = MAX_STATIC_INSTANCES;

    // --- stats --------------------------------------------------------------
    private _stats: FrameStats = {
        frameIndex: 0,
        gpuTimeMs:  0,
        cpuTimeMs:  0,
        realFps:    0,
        drawCalls:  0,
        triangles:  0,
        passCount:  0,
    };

    private _tickStart: number = 0;
    private _lastTickTime: number = 0;
    private _smoothFrameTime: number = 16.67; // EMA of frame-to-frame interval (ms)

    // --- Pooled scratch buffers (avoid per-frame allocations) ----------------
    // Use explicit Float32Array<ArrayBuffer> to satisfy GPUAllowSharedBufferSource typing.
    private _perFrameData        = new Float32Array(new ArrayBuffer(PER_FRAME_BUFFER_SIZE));
    private _blitExposureScratch = new Float32Array(new ArrayBuffer(4));
    private _lightPos3           = new Float32Array(new ArrayBuffer(12));
    private _lightDir3           = new Float32Array(new ArrayBuffer(12));
    private _camPos3             = new Float32Array(new ArrayBuffer(12));
    /** Reusable scratch for static instance matrix packing (MAX_STATIC_INSTANCES * 16 floats). */
    private _instanceScratch:    Float32Array<ArrayBuffer> | null = null;
    /** Reusable scratch for dynamic model matrix packing (MAX_DRAWS * STRIDE_F32 floats). */
    private _modelMatScratch:    Float32Array<ArrayBuffer> | null = null;
    /** Reusable Map for instance batch grouping — cleared each frame instead of reallocated. */
    private _batchGroupMap:      Map<string, { drawables: DrawableRef[]; baseDefines: ShaderDefines; }> = new Map();

    // --- Bind group caches (avoid per-frame createBindGroup allocations) -----
    private _ddBGCache          = new BindGroupCache();   // depth downsample
    private _lightingGBufBGCache = new BindGroupCache();  // deferred lighting G-Buffer BG
    private _fwdSceneBlitBGCache = new BindGroupCache();  // forward scene blit
    private _refrDsBGCache      = new BindGroupCache();   // refraction downsample
    private _blitPresentBGCache = new BindGroupCache();   // present/tonemap blit
    /** Per-variant cache for forward pass @group(0) bind groups. */
    private _fwdPerFrameBGCache: Map<string, BindGroupCache> = new Map();
    /** Skinned shadow: Map<jointBuffer, sparseArray[drawIndex] = bindGroup>. */
    private _skinnedShadowBGPool: Map<GPUBuffer, GPUBindGroup[]> = new Map();
    /** Skinned G-Buffer: Map<variantKey, Map<jointBuffer, sparseArray[drawIndex] = bindGroup>>. */
    private _skinnedGBufBGPool: Map<string, Map<GPUBuffer, GPUBindGroup[]>> = new Map();
    /** Linear-clamp sampler for forward transparent pass (created once, not per-frame). */
    private _forwardLinearSampler: GPUSampler | null = null;

    // --- GPU timestamp query (hardware GPU timing) --------------------------
    private _tsQuerySet:       GPUQuerySet | null = null;
    private _tsResolveBuffer:  GPUBuffer | null = null;
    private _tsReadbackBuffer: GPUBuffer | null = null;
    /** True while a readback mapAsync is in flight — prevents overlapping maps. */
    private _tsReadPending: boolean = false;

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    init(deps: {
        backend:             GPUBackend;
        resources:           ResourceManager;
        shaderSystem:        ShaderSystem;
        pipelineManager:     PipelineManager;
        meshSystem:          MeshSystem;
        renderGraph:         RenderGraph;
        sceneGraph:          SceneGraph;
        cameraSystem:        CameraSystem;
        lightSystem:         LightSystem;
        materialSystem:      MaterialSystem;
        postProcessStack:    PostProcessStack;
        inputSystem?:        InputSystem;
        cameraControllers?:  CameraControllerSystem;
        backgroundSystem?:   BackgroundSystem;
        animationSystem?:    AnimationSystem;
        config?:             EngineConfiguration;
        engine?:             Engine;
    }): void {
        this._backend          = deps.backend;
        this._resources        = deps.resources;
        this._shaderSystem     = deps.shaderSystem;
        this._pipelineManager  = deps.pipelineManager;
        this._meshSystem       = deps.meshSystem;
        this._renderGraph      = deps.renderGraph;
        this._sceneGraph         = deps.sceneGraph;
        this._cameraSystem       = deps.cameraSystem;
        this._lightSystem        = deps.lightSystem;
        this._materialSystem     = deps.materialSystem;
        this._postProcessStack   = deps.postProcessStack;
        this._inputSystem        = deps.inputSystem        ?? null;
        this._cameraControllers  = deps.cameraControllers ?? null;
        this._backgroundSystem   = deps.backgroundSystem  ?? null;
        this._animationSystem    = deps.animationSystem    ?? null;
        this._config             = deps.config             ?? null;
        this._engine             = deps.engine             ?? null;

        // Apply config overrides to rendering limits.
        if (this._config) {
            this._maxDraws           = this._config.maxDrawsPerFrame;
            this._maxStaticInstances = this._config.maxStaticInstances;
        }

        // Pre-allocate pooled scratch buffers at their maximum sizes.
        this._instanceScratch = new Float32Array(new ArrayBuffer(this._maxStaticInstances * 16 * 4));
        this._modelMatScratch = new Float32Array(new ArrayBuffer(this._maxDraws * MODEL_UNIFORM_STRIDE));

        // ---- 1. Per-frame ring buffers (no bind groups yet) -----------------
        //
        // Buffers must exist before GBuffer variant pre-warming, because
        // _getOrCreateGBufferVariant() creates per-variant bind groups that
        // reference these buffers.
        for (let i = 0; i < this._backend.maxFramesInFlight; i++) {
            this._perFrameBuffers.push(this._backend.device.createBuffer({
                label: `PerFrameUniforms[${i}]`,
                size:  PER_FRAME_BUFFER_SIZE,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }));
        }

        // ---- 2. Model matrix buffer (dynamic / per-draw path) ---------------
        this._modelMatBuffer = this._backend.device.createBuffer({
            label: 'ModelMatrixBuffer',
            size:  this._maxDraws * MODEL_UNIFORM_STRIDE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // ---- 2a. Static instance storage buffer (instanced path) ------------
        //
        // Holds up to _maxStaticInstances mat4x4f values (64 bytes each).
        // Written every frame by _buildInstanceBatches(); the shader reads
        // instanceMatrices[instance_index] when USE_INSTANCING=1.
        this._staticInstanceBuffer = this._backend.device.createBuffer({
            label: 'StaticInstanceBuffer',
            size:  this._maxStaticInstances * MODEL_MATRIX_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // ---- 2b. 1×1 white placeholder texture for background bind group ----
        {
            const tex = this._backend.device.createTexture({
                label:  'Placeholder1x1White',
                size:   { width: 1, height: 1 },
                format: 'rgba8unorm',
                usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            this._backend.queue.writeTexture(
                { texture: tex },
                new Uint8Array([255, 255, 255, 255]),
                { bytesPerRow: 4 },
                { width: 1, height: 1 },
            );
            this._placeholder2DView = tex.createView();
        }

        // ---- 3. Compile G-Buffer shader + pre-warm common variants ----------
        //
        // _getOrCreateGBufferVariant() compiles the pipeline AND creates
        // variant-specific per-frame (group 0) and model-matrix (group 2) bind
        // groups, because auto-layout BGLs are pipeline-opaque — two pipelines
        // with identical WGSL group declarations still produce incompatible BGLs.
        this._shaderSystem.registerSource({ label: 'gbuffer', source: gbufferWGSL });
        const gbBase = this._getOrCreateGBufferVariant({});
        this._getOrCreateGBufferVariant({ ALPHA_MASK: '1' });

        // Expose the base-variant per-frame BGL/BGs for external callers
        // (e.g. getPerFrameBindGroup() used by the forward pass placeholder).
        // The base key depends on the current _useDepthPrepass flag.
        const baseKey = 'dp|';
        this._perFrameBindGroupLayout = gbBase.getBindGroupLayout(0);
        for (const bg of this._gbufferPerFrameBGCache.get(baseKey)!) {
            this._perFrameBindGroups.push(bg);
        }

        // ---- 1a. Depth prepass pipeline + dedicated bind groups -------------
        //
        // The depth prepass shader uses @group(0) (per-frame) and @group(2)
        // (model matrix) with auto-derived BGLs that are opaque to other
        // pipelines.  We create separate bind groups wrapping the same GPU
        // buffers so they are compatible with this pipeline's auto layout.
        this._shaderSystem.registerSource({ label: 'depth_prepass', source: depthPrepassWGSL });
        const dpVariant = this._shaderSystem.getVariant('depth_prepass', {});

        this._depthPrepassPipeline = this._pipelineManager.getOrCreateDepthPrepassPipeline(
            dpVariant.handle,
            STANDARD_VERTEX_LAYOUT,
        );

        const dpPerFrameBGL = this._depthPrepassPipeline.getBindGroupLayout(0);
        const dpModelBGL    = this._depthPrepassPipeline.getBindGroupLayout(2);

        for (let i = 0; i < this._backend.maxFramesInFlight; i++) {
            this._depthPrepassPerFrameBindGroups.push(
                this._backend.device.createBindGroup({
                    label:   `DepthPrepass/PerFrame/BG[${i}]`,
                    layout:  dpPerFrameBGL,
                    entries: [{ binding: 0, resource: { buffer: this._perFrameBuffers[i]! } }],
                }),
            );
        }

        for (let i = 0; i < this._maxDraws; i++) {
            this._depthPrepassModelMatBindGroups.push(
                this._backend.device.createBindGroup({
                    label:   `DepthPrepass/ModelMatrix/BG[${i}]`,
                    layout:  dpModelBGL,
                    entries: [{
                        binding:  0,
                        resource: {
                            buffer: this._modelMatBuffer!,
                            offset: i * MODEL_UNIFORM_STRIDE,
                            size:   MODEL_MATRIX_SIZE,
                        },
                    }],
                }),
            );
        }

        // ---- 1b. Instanced depth prepass pipeline + bind groups -------------
        //
        // USE_INSTANCING=1 variant of the depth prepass: group(2) is a storage
        // buffer instead of a uniform sub-range.  Per-frame group(0) uses the
        // same underlying GPU buffers but must be bound against THIS pipeline's
        // auto-derived BGL (different pipeline → different BGL object).
        const dpInstVariant = this._shaderSystem.getVariant('depth_prepass', { USE_INSTANCING: '1' });

        this._depthPrepassInstancedPipeline = this._pipelineManager.getOrCreateDepthPrepassPipeline(
            dpInstVariant.handle,
            STANDARD_VERTEX_LAYOUT,
        );

        const dpInstPerFrameBGL = this._depthPrepassInstancedPipeline.getBindGroupLayout(0);
        const dpInstModelBGL    = this._depthPrepassInstancedPipeline.getBindGroupLayout(2);

        for (let i = 0; i < this._backend.maxFramesInFlight; i++) {
            this._depthPrepassInstancedPerFrameBGs.push(
                this._backend.device.createBindGroup({
                    label:   `DepthPrepass/Instanced/PerFrame/BG[${i}]`,
                    layout:  dpInstPerFrameBGL,
                    entries: [{ binding: 0, resource: { buffer: this._perFrameBuffers[i]! } }],
                }),
            );
        }

        this._depthPrepassInstancedG2BG = this._backend.device.createBindGroup({
            label:   'DepthPrepass/Instanced/G2/BG',
            layout:  dpInstModelBGL,
            entries: [{ binding: 0, resource: { buffer: this._staticInstanceBuffer! } }],
        });

        // ---- 4. Deferred lighting pipeline ---------------------------------
        this._shaderSystem.registerSource({ label: 'deferred_lighting', source: deferredLightingWGSL });
        const dlVariant = this._shaderSystem.getVariant('deferred_lighting', {});

        this._lightingPipeline = this._pipelineManager.getOrCreateDeferredLightingPipeline(
            dlVariant.handle,
            dlVariant.handle,
        );

        // Derive BGLs from the lighting pipeline (layout: 'auto').
        this._lightingGBufferBGL = this._lightingPipeline.getBindGroupLayout(1);
        this._lightingLightsBGL  = this._lightingPipeline.getBindGroupLayout(2);

        // Per-frame bind groups for the lighting pipeline: same buffers as the
        // G-Buffer ring, but bound against THIS pipeline's @group(0) BGL.
        const lightingPerFrameBGL = this._lightingPipeline.getBindGroupLayout(0);
        for (let i = 0; i < this._backend.maxFramesInFlight; i++) {
            this._lightingPerFrameBindGroups.push(
                this._backend.device.createBindGroup({
                    label:   `Lighting/PerFrame/BG[${i}]`,
                    layout:  lightingPerFrameBGL,
                    entries: [{ binding: 0, resource: { buffer: this._perFrameBuffers[i]! } }],
                }),
            );
        }

        // Nearest-clamp sampler for G-Buffer texture reads in the lighting pass.
        this._clampSampler = this._backend.device.createSampler({
            label:        'GBuffer/ClampSampler',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            magFilter:    'nearest',
            minFilter:    'nearest',
        });

        // Linear-clamp sampler for forward transparent pass refraction sampling.
        this._forwardLinearSampler = this._backend.device.createSampler({
            label:        'Forward/LinearClampSampler',
            magFilter:    'linear',
            minFilter:    'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
        });

        // PCF comparison sampler for shadow atlas reads.
        this._shadowCompSampler = this._backend.device.createSampler({
            label:        'Shadow/CompSampler',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            magFilter:    'linear',
            minFilter:    'linear',
            compare:      'less',
        });

        // Lighting @group(2): light buffer + shadow atlas + comparison sampler + shadow data.
        // LightSystem resources are always created in LightSystem.init() regardless of scene.
        const lightGpuBuffer   = this._resources.getBuffer(this._lightSystem.getLightStorageBuffer());
        const shadowAtlasGPU   = this._resources.getTexture(this._lightSystem.getShadowAtlasTexture());
        const shadowDataGpuBuf = this._resources.getBuffer(this._lightSystem.getShadowDataBuffer());

        if (lightGpuBuffer && shadowAtlasGPU && shadowDataGpuBuf && this._lightingLightsBGL) {
            this._lightingLightsBG = this._backend.device.createBindGroup({
                label:   'Lighting/Lights/BG',
                layout:  this._lightingLightsBGL,
                entries: [
                    { binding: 0, resource: { buffer: lightGpuBuffer } },
                    { binding: 1, resource: shadowAtlasGPU.createView({ aspect: 'depth-only' }) },
                    { binding: 2, resource: this._shadowCompSampler! },
                    { binding: 3, resource: { buffer: shadowDataGpuBuf } },
                ],
            });
        }

        // ---- 4a-env. Environment cubemap @group(3) for IBL reflections ------
        //
        // TODO: When reflection probes are implemented, this bind group should
        //       be updated per-object with the closest probe's cubemap instead
        //       of the single scene-level environment cubemap.

        this._lightingEnvBGL = this._lightingPipeline.getBindGroupLayout(3);

        // EnvParams uniform buffer (16 bytes: u32 enabled + 3× u32 padding)
        this._envParamsBuffer = this._backend.device.createBuffer({
            label: 'Lighting/EnvParams',
            size:  16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // Initially disabled (enabled = 0)
        this._uploadEnvParams(0);

        // 1×1 placeholder cubemap (6 faces, black)
        this._placeholderCubeTex = this._backend.device.createTexture({
            label:  'Placeholder/Cubemap',
            size:   { width: 1, height: 1, depthOrArrayLayers: 6 },
            format: 'rgba8unorm',
            usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        const blackPixel = new Uint8Array([0, 0, 0, 255]);
        for (let face = 0; face < 6; face++) {
            this._backend.queue.writeTexture(
                { texture: this._placeholderCubeTex, origin: { x: 0, y: 0, z: face } },
                blackPixel, { bytesPerRow: 4 }, { width: 1, height: 1 },
            );
        }

        this._envCubeSampler = this._backend.device.createSampler({
            label:        'Env/CubeSampler',
            magFilter:    'linear',
            minFilter:    'linear',
            mipmapFilter: 'linear',
        });

        // Build initial @group(3) with the placeholder cubemap (disabled)
        this._lightingEnvBG = this._backend.device.createBindGroup({
            label:  'Lighting/Env/BG',
            layout: this._lightingEnvBGL,
            entries: [
                { binding: 0, resource: { buffer: this._envParamsBuffer } },
                { binding: 1, resource: this._placeholderCubeTex.createView({ dimension: 'cube' }) },
                { binding: 2, resource: this._envCubeSampler },
            ],
        });

        // ---- 4b. Shadow pipeline + per-cascade / per-draw bind groups -------
        this._shaderSystem.registerSource({ label: 'shadow', source: shadowWGSL });

        const shadowVariant = this._shaderSystem.getVariant('shadow', {});
        this._shadowPipeline = this._pipelineManager.getOrCreateShadowPipeline(
            shadowVariant.handle, STANDARD_VERTEX_LAYOUT,
        );

        const shadowInstVariant = this._shaderSystem.getVariant('shadow', { USE_INSTANCING: '1' });
        this._shadowInstancedPipeline = this._pipelineManager.getOrCreateShadowPipeline(
            shadowInstVariant.handle, STANDARD_VERTEX_LAYOUT,
        );

        const shadowSkinnedVariant = this._shaderSystem.getVariant('shadow', { USE_SKINNING: '1' });
        this._shadowSkinnedPipeline = this._pipelineManager.getOrCreateShadowPipeline(
            shadowSkinnedVariant.handle, SKINNED_VERTEX_LAYOUT,
        );

        const shadowVPBGL     = this._shadowPipeline.getBindGroupLayout(0);
        const shadowVPInstBGL = this._shadowInstancedPipeline.getBindGroupLayout(0);
        const shadowVPSkinnedBGL = this._shadowSkinnedPipeline.getBindGroupLayout(0);
        const shadowModelBGL  = this._shadowPipeline.getBindGroupLayout(2);
        const shadowInstG2BGL = this._shadowInstancedPipeline.getBindGroupLayout(2);

        for (let ci = 0; ci < MAX_SHADOW_CASCADE_SLOTS; ci++) {
            const vpBuf = this._backend.device.createBuffer({
                label: `Shadow/VP[${ci}]`,
                size:  SHADOW_VP_BUFFER_SIZE,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            this._shadowVPBuffers.push(vpBuf);
            this._shadowVPBGs.push(this._backend.device.createBindGroup({
                label:   `Shadow/VP/BG[${ci}]`,
                layout:  shadowVPBGL,
                entries: [{ binding: 0, resource: { buffer: vpBuf } }],
            }));
            this._shadowVPInstBGs.push(this._backend.device.createBindGroup({
                label:   `Shadow/VP/Inst/BG[${ci}]`,
                layout:  shadowVPInstBGL,
                entries: [{ binding: 0, resource: { buffer: vpBuf } }],
            }));
            this._shadowVPSkinnedBGs.push(this._backend.device.createBindGroup({
                label:   `Shadow/VP/Skinned/BG[${ci}]`,
                layout:  shadowVPSkinnedBGL,
                entries: [{ binding: 0, resource: { buffer: vpBuf } }],
            }));
        }

        for (let i = 0; i < this._maxDraws; i++) {
            this._shadowModelMatBGs.push(this._backend.device.createBindGroup({
                label:   `Shadow/ModelMatrix/BG[${i}]`,
                layout:  shadowModelBGL,
                entries: [{
                    binding:  0,
                    resource: {
                        buffer: this._modelMatBuffer!,
                        offset: i * MODEL_UNIFORM_STRIDE,
                        size:   MODEL_MATRIX_SIZE,
                    },
                }],
            }));
        }

        this._shadowInstancedG2BG = this._backend.device.createBindGroup({
            label:   'Shadow/Instanced/G2/BG',
            layout:  shadowInstG2BGL,
            entries: [{ binding: 0, resource: { buffer: this._staticInstanceBuffer! } }],
        });

        // ---- 4c. Transparent shadow pipeline + bind groups -------------------
        this._shaderSystem.registerSource({ label: 'shadow_transparent', source: shadowTransparentWGSL });
        {
            const stVariant = this._shaderSystem.getVariant('shadow_transparent', {});
            const stPipeline = this._pipelineManager.getOrCreateRenderPipeline({
                vertexShader:        stVariant.handle,
                fragmentShader:      stVariant.handle,
                vertexBufferLayouts: buildVertexBufferLayouts(STANDARD_VERTEX_LAYOUT),
                colorTargetFormats:  [],
                depthStencilFormat:  'depth32float',
                cullMode:            'back',
                depthWriteEnabled:   true,
                depthCompare:        'less',
            });
            this._shadowTransparentPipelineCache.set('', stPipeline);

            const stVPBGL    = stPipeline.getBindGroupLayout(0);
            const stModelBGL = stPipeline.getBindGroupLayout(2);

            for (let ci = 0; ci < MAX_SHADOW_CASCADE_SLOTS; ci++) {
                this._shadowVPTransparentBGs.push(this._backend.device.createBindGroup({
                    label:   `Shadow/VP/Transparent/BG[${ci}]`,
                    layout:  stVPBGL,
                    entries: [{ binding: 0, resource: { buffer: this._shadowVPBuffers[ci]! } }],
                }));
            }

            for (let i = 0; i < this._maxDraws; i++) {
                this._shadowTransparentModelMatBGs.push(this._backend.device.createBindGroup({
                    label:   `Shadow/Transparent/ModelMatrix/BG[${i}]`,
                    layout:  stModelBGL,
                    entries: [{
                        binding:  0,
                        resource: {
                            buffer: this._modelMatBuffer!,
                            offset: i * MODEL_UNIFORM_STRIDE,
                            size:   MODEL_MATRIX_SIZE,
                        },
                    }],
                }));
            }
        }

        // ---- 5. Blit pipeline (tonemapping → swap-chain) --------------------
        this._shaderSystem.registerSource({ label: 'blit', source: blitWGSL });
        const blitVariant = this._shaderSystem.getVariant('blit', {});

        this._blitPipeline = this._pipelineManager.getOrCreateFullscreenPipeline(
            blitVariant.handle,
            blitVariant.handle,
            this._backend.preferredFormat,
        );
        this._blitBGL = this._blitPipeline.getBindGroupLayout(0);

        // Fullscreen copy pipeline (HDR → HDR, no tonemapping).
        this._shaderSystem.registerSource({ label: 'fullscreen_copy', source: fullscreenCopyWGSL });
        const copyVariant = this._shaderSystem.getVariant('fullscreen_copy', {});
        this._fullscreenCopyPipeline = this._pipelineManager.getOrCreateFullscreenPipeline(
            copyVariant.handle, copyVariant.handle, 'rgba16float',
        );

        this._blitExposureBuffer = this._backend.device.createBuffer({
            label: 'Blit/ExposureUniform',
            size:  4, // single f32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // ---- 5b. Depth downsample compute pipeline ----------------------------
        this._shaderSystem.registerSource({ label: 'depth_downsample', source: depthDownsampleWGSL });
        const ddVariant = this._shaderSystem.getVariant('depth_downsample', {});
        this._depthDownsamplePipeline = this._pipelineManager.getOrCreateComputePipeline({
            computeShader: ddVariant.handle,
            entryPoint:    'cs_main',
        });

        // ---- 5c. Forward transparent pipeline + bind groups --------------------
        this._shaderSystem.registerSource({ label: 'forward_transparent', source: forwardTransparentWGSL });
        {
            const fwdVariant = this._shaderSystem.getVariant('forward_transparent', {});
            const fwdPipeline = this._pipelineManager.getOrCreateForwardPipeline(
                fwdVariant.handle, fwdVariant.handle, STANDARD_VERTEX_LAYOUT,
                { blend: BLEND_STRAIGHT_ALPHA, doubleSided: false },
            );
            this._forwardPipelineCache.set('', fwdPipeline);

            // @group(0) per-frame bind groups are created per-frame in the
            // forward pass callback because they include the scene color copy
            // texture which changes each frame.

            // @group(2) model matrix bind groups (one per draw slot)
            const fwdModelBGL = fwdPipeline.getBindGroupLayout(2);
            for (let i = 0; i < this._maxDraws; i++) {
                this._forwardModelMatBGs.push(
                    this._backend.device.createBindGroup({
                        label:   `Forward/ModelMatrix/BG[${i}]`,
                        layout:  fwdModelBGL,
                        entries: [{
                            binding:  0,
                            resource: {
                                buffer: this._modelMatBuffer!,
                                offset: i * MODEL_UNIFORM_STRIDE,
                                size:   MODEL_MATRIX_SIZE,
                            },
                        }],
                    }),
                );
            }

            // @group(3) lights + shadows + env (combined bind group)
            this._forwardLightEnvBGL = fwdPipeline.getBindGroupLayout(3);
            const fwdLightEnvBGL = this._forwardLightEnvBGL;
            if (lightGpuBuffer && shadowAtlasGPU && shadowDataGpuBuf) {
                this._forwardLightEnvBG = this._backend.device.createBindGroup({
                    label:   'Forward/LightEnv/BG',
                    layout:  fwdLightEnvBGL,
                    entries: [
                        { binding: 0, resource: { buffer: lightGpuBuffer } },
                        { binding: 1, resource: shadowAtlasGPU.createView({ aspect: 'depth-only' }) },
                        { binding: 2, resource: this._shadowCompSampler! },
                        { binding: 3, resource: { buffer: shadowDataGpuBuf } },
                        { binding: 4, resource: { buffer: this._envParamsBuffer! } },
                        { binding: 5, resource: this._placeholderCubeTex!.createView({ dimension: 'cube' }) },
                        { binding: 6, resource: this._envCubeSampler! },
                    ],
                });
            }
        }

        // ---- 6. GPU timestamp queries (if supported) --------------------------
        if (this._backend.hasTimestampQuery) {
            this._tsQuerySet = this._backend.device.createQuerySet({
                type: 'timestamp',
                count: 2,
            });
            // Resolve buffer: 2 × uint64 = 16 bytes
            this._tsResolveBuffer = this._backend.device.createBuffer({
                label: 'TimestampResolve',
                size: 16,
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
            });
            // Readback buffer: mappable for CPU read
            this._tsReadbackBuffer = this._backend.device.createBuffer({
                label: 'TimestampReadback',
                size: 16,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            this._renderGraph.setTimestampQuery(this._tsQuerySet, this._tsResolveBuffer);
            this._log.info('GPU timestamp queries enabled');
        }

        // ---- 7. Post-process stack init ----------------------------------------
        this._postProcessStack.init(this._buildPostProcessContext());

        this._log.info(
            `Initialized — ${this._backend.maxFramesInFlight} per-frame ring slots, ` +
            `${this._maxDraws} dynamic model-matrix slots (${this._maxDraws * MODEL_UNIFORM_STRIDE / 1024} KB), ` +
            `${this._maxStaticInstances} static instance slots (${this._maxStaticInstances * MODEL_MATRIX_SIZE / 1024} KB)`,
        );
    }

    // -------------------------------------------------------------------------
    // Main loop control
    // -------------------------------------------------------------------------

    start(): void {
        if (this._running) return;
        this._running = true;
        this._lastTimestamp = performance.now();
        this._rafId = requestAnimationFrame(this._tick.bind(this));
    }

    stop(): void {
        this._running = false;
        cancelAnimationFrame(this._rafId);
    }

    // -------------------------------------------------------------------------
    // Frame loop
    // -------------------------------------------------------------------------

    private _tick(timestamp: number): void {
        if (!this._running) return;

        const now = performance.now();
        this._tickStart = now;
        const dt = (timestamp - this._lastTimestamp) / 1000;
        this._lastTimestamp = timestamp;

        // Real FPS from actual frame-to-frame interval (EMA smoothed).
        if (this._lastTickTime > 0) {
            const frameMs = now - this._lastTickTime;
            this._smoothFrameTime = this._smoothFrameTime * 0.9 + frameMs * 0.1;
            this._stats.realFps = 1000 / this._smoothFrameTime;
        }
        this._lastTickTime = now;

        this._beginFrame();
        this._updateScene(dt);
        this._buildRenderGraph();
        this._compileAndExecute();

        const cpuTimeMs = performance.now() - this._tickStart;

        // --- GPU timing readback ---
        if (this._tsQuerySet && this._tsResolveBuffer && this._tsReadbackBuffer) {
            // Copy + map only when no read is already in flight.
            // The readback buffer cannot be used in a submit while mapped.
            if (!this._tsReadPending) {
                const copyEncoder = this._backend.device.createCommandEncoder({ label: 'TimestampCopy' });
                copyEncoder.copyBufferToBuffer(this._tsResolveBuffer, 0, this._tsReadbackBuffer, 0, 16);
                this._backend.queue.submit([copyEncoder.finish()]);

                this._tsReadPending = true;
                this._tsReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
                    const data = new BigUint64Array(this._tsReadbackBuffer!.getMappedRange());
                    const begin = data[0]!;
                    const end   = data[1]!;
                    this._tsReadbackBuffer!.unmap();
                    this._tsReadPending = false;
                    // Timestamps are in nanoseconds.
                    if (end > begin) {
                        this._stats.gpuTimeMs = Number(end - begin) / 1_000_000;
                    }
                }).catch(() => {
                    this._tsReadPending = false;
                });
            }
        } else {
            // Fallback: measure from just after submit (not tick start) to
            // onSubmittedWorkDone. This still includes promise scheduling
            // jitter but at least excludes CPU encoding time.
            const submitTime = performance.now();
            this._backend.queue.onSubmittedWorkDone().then(() => {
                this._stats.gpuTimeMs = performance.now() - submitTime;
            });
        }

        this._endFrame(cpuTimeMs);

        this._rafId = requestAnimationFrame(this._tick.bind(this));
    }

    // -------------------------------------------------------------------------
    // Frame phases
    // -------------------------------------------------------------------------

    private _beginFrame(): void {
        this._frameIndex++;
        this._resources.beginFrame(this._frameIndex);
        this._renderGraph.reset();
        this._stats.drawCalls = 0;
        this._stats.triangles = 0;

        // Reset per-frame mouse deltas
        this._inputSystem?.beginFrame();
    }

    private _updateScene(dt: number): void {
        // 0. Camera controller — moves the camera node before world matrices are propagated.
        if (this._inputSystem && this._cameraControllers) {
            this._cameraControllers.update(dt, this._inputSystem.getState());
        }

        // 0b. Advance animations — evaluates keyframes and writes new TRS to nodes.
        //     Must happen before world-matrix propagation so animated joint nodes
        //     have up-to-date local transforms.
        if (this._animationSystem) {
            this._animationSystem.updateAnimations(dt);
        }

        // 0c. User update callbacks — run after animations so user code can read
        //     animated values and override/augment them before world matrices propagate.
        if (this._engine) {
            this._engine._runUpdateCallbacks(dt, this._frameIndex);
        }

        // 1. Propagate world matrices through the transform hierarchy.
        this._sceneGraph.updateWorldMatrices();

        // 1b. Compute skinning joint matrices from the updated world matrices
        //      and upload them to the GPU before any draw calls.
        if (this._animationSystem) {
            this._animationSystem.computeJointMatrices();
            this._animationSystem.uploadJointMatrices();
        }

        // 2. Recompute active camera matrices.
        const activeCamera = this._cameraSystem.getActiveCamera();
        const cameraPosition = this._camPos3;
        cameraPosition[0] = 0; cameraPosition[1] = 0; cameraPosition[2] = 0;

        if (activeCamera) {
            const camNode = activeCamera.nodeHandle !== null
                ? this._sceneGraph.getNode(activeCamera.nodeHandle)
                : undefined;
            const worldMatrix = camNode?.worldMatrix ?? _identityMat4();

            // Extract camera world position from column 3 of the world matrix.
            // Column-major layout: translation at indices [12, 13, 14].
            cameraPosition[0] = worldMatrix[12]!;
            cameraPosition[1] = worldMatrix[13]!;
            cameraPosition[2] = worldMatrix[14]!;

            this._cameraSystem.update(activeCamera.handle, worldMatrix);
            this._cameraSystem.advanceTAAJitter(activeCamera.handle);
        }

        // 3. Frustum cull with pre-extracted planes + bounding spheres from MeshSystem.
        const cam = this._cameraSystem.getActiveCamera();
        if (cam) {
            // Pass empty planes array when frustum culling is disabled.
            const useFrustumCulling = this._config?.frustumCulling ?? true;
            const tolerance = this._config?.frustumCullTolerance ?? 0;
            this._cullResults = this._sceneGraph.cull(
                cam.viewProjectionMatrix,
                cameraPosition,
                useFrustumCulling ? cam.frustumPlanes : [],
                (h) => this._meshSystem.getBoundingSphere(h),
                tolerance,
                (mh) => this._materialSystem.getRenderPath(mh) === RenderPath.Forward,
            );
        }

        // 3a. Sort opaque drawables into static instance batches and dynamic list.
        //     Must run before _packModelMatrices() which reads _dynamicDrawables.
        this._buildInstanceBatches();

        // 3a-2. Sort dynamic drawables by materialHandle to minimize pipeline switches
        //        in the GBuffer pass (same material → same shader variant → no setPipeline call).
        this._dynamicDrawables.sort((a, b) => a.materialHandle - b.materialHandle);

        // 3b. Upload dynamic world matrices into the per-draw uniform ring buffer.
        this._packModelMatrices();

        // 3c. Upload transparent world matrices after opaque draws.
        this._packTransparentMatrices();

        // 4. Push world transforms from scene nodes into each light record.
        const lPos = this._lightPos3;
        const lDir = this._lightDir3;
        for (const lightHandle of this._cullResults.visibleLights) {
            const lightNode = this._sceneGraph.getNode(lightHandle);
            if (!lightNode) continue;
            const m = lightNode.worldMatrix;
            // Translation: column 3 of column-major mat4.
            lPos[0] = m[12]!; lPos[1] = m[13]!; lPos[2] = m[14]!;
            // Forward direction: -Z column (column 2, negated) for a standard camera-like transform.
            lDir[0] = -m[8]!; lDir[1] = -m[9]!; lDir[2] = -m[10]!;
            if (lightNode.lightHandle !== undefined) {
                this._lightSystem.setLightTransform(lightNode.lightHandle, lPos, lDir);
            }
        }

        // 4b. Recompute shadow matrices (must run after setLightTransform).
        if (cam) {
            const camNear = cam.perspective?.near ?? cam.orthographic?.near ?? 0.1;
            const camFar  = cam.perspective?.far  ?? cam.orthographic?.far  ?? 1000;
            for (const lightHandle of this._lightSystem.getShadowCasters()) {
                this._lightSystem.computeShadowMatrices(lightHandle, {
                    viewMatrix: cam.viewMatrix,
                    projMatrix: cam.projectionMatrix,
                    near:       camNear,
                    far:        camFar,
                });
            }
        }

        // 5. Rebuild clustered light bins.
        if (cam) {
            this._lightSystem.buildClusters(
                cam.viewMatrix,
                cam.projectionMatrix,
            );
        }

        // 6. Upload dirty material uniform buffers.
        this._materialSystem.uploadDirtyMaterials();

        // 7. Upload per-light structured buffer + shadow data.
        this._lightSystem.uploadLightData();
        this._lightSystem.uploadShadowData();

        // 8. Cache inverse VP for the background cubemap pass.
        const invVP = cam?.inverseViewProjection ?? _identityMat4();
        this._lastInvViewProj.set(invVP);

        // 9. Pack and write per-frame uniforms into the current ring slot.
        const canvas = this._backend.context.canvas as HTMLCanvasElement;
        this._uploadPerFrameUniforms({
            viewMatrix:            cam?.viewMatrix            ?? _identityMat4(),
            projectionMatrix:      cam?.projectionMatrix      ?? _identityMat4(),
            viewProjectionMatrix:  cam?.viewProjectionMatrix  ?? _identityMat4(),
            inverseViewProjection: cam?.inverseViewProjection ?? _identityMat4(),
            cameraPosition,

            time:       performance.now() / 1000,
            deltaTime:  dt,
            resolution: [canvas.width, canvas.height],
            frameIndex: this._frameIndex,
            exposure:   cam ? this._cameraSystem.getExposure(cam.handle) : 0,
            jitter:     cam ? [cam.jitterX, cam.jitterY] : [0, 0],
        });

        // Write pre-computed linear exposure multiplier to the blit uniform buffer.
        // Avoids exp2() per pixel in the shader.
        if (this._blitExposureBuffer) {
            const ev100 = cam ? this._cameraSystem.getExposure(cam.handle) : 0;
            const linearExposure = 1.0 / (Math.pow(2, ev100) * 1.2);
            this._blitExposureScratch[0] = linearExposure;
            this._backend.queue.writeBuffer(this._blitExposureBuffer, 0, this._blitExposureScratch);
        }
    }

    /**
     * Separate opaque drawables into:
     *   - static instance batches  → packed into _staticInstanceBuffer
     *   - dynamic drawables        → packed into the per-draw uniform buffer
     *
     * Static objects that share the same mesh + material are merged into a
     * single InstanceBatch and rendered with drawIndexed(…, instanceCount, …).
     * Dynamic objects (isStatic=false) keep the existing one-draw-per-object path.
     *
     * TODO (Animations): skeletal / morph-target / keyframe animated nodes must
     *   set isStatic=false so they remain on the dynamic path.  When animation
     *   systems are implemented they should call scene.setNodeStatic(handle, false)
     *   for every node they control.
     */
    private _buildInstanceBatches(): void {
        this._instanceBatches.length  = 0;
        this._dynamicDrawables.length = 0;

        if (!this._staticInstanceBuffer) {
            // Fallback: treat everything as dynamic if the buffer is missing.
            for (const d of this._cullResults.opaqueDrawables) {
                this._dynamicDrawables.push(d);
            }
            return;
        }

        // Group static drawables by (meshHandle, materialHandle).
        // Reuse the Map — clear it instead of reallocating.
        const groups = this._batchGroupMap;
        groups.clear();

        for (const d of this._cullResults.opaqueDrawables) {
            if (!d.isStatic) {
                this._dynamicDrawables.push(d);
                continue;
            }
            const key = `${d.meshHandle}|${d.materialHandle}`;
            let group = groups.get(key);
            if (!group) {
                const matRecord = this._materialSystem.getMaterial(d.materialHandle);
                group = { drawables: [], baseDefines: matRecord?.shaderDefines ?? {} };
                groups.set(key, group);
            }
            group.drawables.push(d);
        }

        const staticCount = this._cullResults.opaqueDrawables.length - this._dynamicDrawables.length;
        if (staticCount === 0) return;

        const capped = Math.min(staticCount, this._maxStaticInstances);

        // Pack world matrices and build batch records (reuse pooled scratch buffer).
        const scratch = this._instanceScratch!;
        let offset = 0;

        for (const [, group] of groups) {
            if (offset >= capped) break;
            const firstInstance = offset;
            for (const d of group.drawables) {
                if (offset >= capped) break;
                scratch.set(d.worldMatrix, offset * 16);
                offset++;
            }
            const batchCount = offset - firstInstance;
            this._instanceBatches.push({
                meshHandle:     group.drawables[0]!.meshHandle,
                materialHandle: group.drawables[0]!.materialHandle,
                baseDefines:    group.baseDefines,
                firstInstance,
                instanceCount:  batchCount,
            });
        }

        this._backend.queue.writeBuffer(
            this._staticInstanceBuffer, 0,
            scratch.buffer, 0, offset * MODEL_MATRIX_SIZE,
        );
    }

    /**
     * Pack world matrices of DYNAMIC (non-static) opaque drawables into the
     * per-draw uniform ring buffer.  Each slot is MODEL_UNIFORM_STRIDE bytes;
     * only the first MODEL_MATRIX_SIZE bytes (mat4x4f) are read by the shader.
     */
    private _packModelMatrices(): void {
        const drawables = this._dynamicDrawables;
        const count = Math.min(drawables.length, this._maxDraws);
        if (count === 0 || !this._modelMatBuffer) return;

        // Stride is 64 f32 elements (256 bytes); only write the first 16 (mat4).
        const STRIDE_F32 = MODEL_UNIFORM_STRIDE / 4;
        const scratch = this._modelMatScratch!;
        for (let i = 0; i < count; i++) {
            scratch.set(drawables[i]!.worldMatrix, i * STRIDE_F32);
        }
        // Write only the filled portion (count slots).
        this._backend.queue.writeBuffer(
            this._modelMatBuffer, 0,
            scratch.buffer, 0, count * MODEL_UNIFORM_STRIDE,
        );
    }

    /**
     * Pack world matrices of transparent drawables into the model-matrix
     * ring buffer immediately after the opaque dynamic slots.
     * Runs once per frame so both the shadow pass and the forward pass
     * can reference these slots.
     */
    private _packTransparentMatrices(): void {
        const transparents = this._cullResults.transparentDrawables;
        const opaqueCount  = this._dynamicDrawables.length;
        const maxTransparent = Math.min(transparents.length, this._maxDraws - opaqueCount);
        if (maxTransparent <= 0 || !this._modelMatBuffer) return;

        const STRIDE_F32 = MODEL_UNIFORM_STRIDE / 4;
        const scratch = this._modelMatScratch!;
        for (let i = 0; i < maxTransparent; i++) {
            scratch.set(transparents[i]!.worldMatrix, (opaqueCount + i) * STRIDE_F32);
        }
        this._backend.queue.writeBuffer(
            this._modelMatBuffer, opaqueCount * MODEL_UNIFORM_STRIDE,
            scratch.buffer, opaqueCount * MODEL_UNIFORM_STRIDE,
            maxTransparent * MODEL_UNIFORM_STRIDE,
        );
    }

    private _buildRenderGraph(): void {
        const canvas = this._backend.context.canvas as HTMLCanvasElement;
        const w = Math.max(1, canvas.width);
        const h = Math.max(1, canvas.height);
        const COLOR = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
        const DEPTH = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;

        // ---- Declare virtual G-Buffer textures ------------------------------

        const gbAlbedoAO = this._renderGraph.declareTexture('GBuffer/AlbedoAO', {
            size: [w, h], format: 'rgba8unorm',   usage: COLOR,
        });
        const gbNormalRoughness = this._renderGraph.declareTexture('GBuffer/NormalRoughness', {
            size: [w, h], format: 'rgba16float',  usage: COLOR,
        });
        const gbMetallicEmissive = this._renderGraph.declareTexture('GBuffer/MetallicEmissive', {
            size: [w, h], format: 'rgba8unorm',   usage: COLOR,
        });
        const depth = this._renderGraph.declareTexture('Depth', {
            size: [w, h], format: 'depth32float', usage: DEPTH,
        });
        const hdrColor = this._renderGraph.declareTexture('HDR/Color', {
            size: [w, h], format: 'rgba16float',  usage: COLOR,
        });

        // ---- Shadow passes --------------------------------------------------

        const shadowCasters     = this._lightSystem.getShadowCasters();
        const shadowAtlasHandle = this._lightSystem.getShadowAtlasTexture();

        if (shadowCasters.length > 0 && shadowAtlasHandle !== 0) {
            const shadowAtlasVId    = this._renderGraph.importTexture('ShadowAtlas', shadowAtlasHandle);
            const shadowCasterInfos = this._lightSystem.getShadowCasterInfos();

            // One render pass covers all lights + cascades.
            // The atlas is cleared once; viewport/scissor selects each cascade's sub-region.
            this._renderGraph.addShadowPass(
                { resourceId: shadowAtlasVId, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1.0 },
                (_enc, passEncoder) => {
                    if (!passEncoder || !this._shadowPipeline || !this._shadowInstancedPipeline) return;
                    const rp = passEncoder as GPURenderPassEncoder;

                    let cascadeSlot = 0;
                    for (const casterInfo of shadowCasterInfos) {
                        for (const cascade of casterInfo.cascades) {
                            if (cascadeSlot >= MAX_SHADOW_CASCADE_SLOTS) break;

                            const region = cascade.atlasRegion;
                            if (region.w <= 0 || region.h <= 0) { cascadeSlot++; continue; }

                            // Upload this cascade's VP matrix to its dedicated uniform buffer.
                            const vpBuf = this._shadowVPBuffers[cascadeSlot];
                            if (!vpBuf) { cascadeSlot++; continue; }
                            this._backend.queue.writeBuffer(
                                vpBuf, 0,
                                cascade.viewProj.buffer as ArrayBuffer,
                                cascade.viewProj.byteOffset,
                                SHADOW_VP_BUFFER_SIZE,
                            );

                            // Restrict rendering to this cascade's atlas sub-region.
                            rp.setViewport(region.x, region.y, region.w, region.h, 0, 1);
                            rp.setScissorRect(region.x, region.y, region.w, region.h);

                            // ── Instanced static batches ──────────────────────
                            if (this._instanceBatches.length > 0 && this._shadowInstancedG2BG) {
                                rp.setPipeline(this._shadowInstancedPipeline);
                                rp.setBindGroup(0, this._shadowVPInstBGs[cascadeSlot]!);
                                rp.setBindGroup(2, this._shadowInstancedG2BG);

                                for (const batch of this._instanceBatches) {
                                    const drawData = this._meshSystem.getDrawData(batch.meshHandle);
                                    if (!drawData) continue;
                                    for (let j = 0; j < drawData.vertexBuffers.length; j++) {
                                        rp.setVertexBuffer(j, drawData.vertexBuffers[j]!);
                                    }
                                    const lod0 = drawData.lodLevels[0];
                                    if (!lod0) continue;
                                    for (const subMesh of lod0.subMeshes) {
                                        if (drawData.indexBuffer && subMesh.indexCount > 0) {
                                            rp.setIndexBuffer(drawData.indexBuffer, drawData.indexFormat);
                                            rp.drawIndexed(subMesh.indexCount, batch.instanceCount,
                                                subMesh.indexOffset, subMesh.vertexOffset, batch.firstInstance);
                                        } else if (subMesh.vertexCount > 0) {
                                            rp.draw(subMesh.vertexCount, batch.instanceCount,
                                                subMesh.vertexOffset, batch.firstInstance);
                                        }
                                    }
                                }
                            }

                            // ── Per-draw dynamic objects ──────────────────────
                            const dynCount = Math.min(this._dynamicDrawables.length, this._maxDraws);
                            if (dynCount > 0) {
                                let currentPipelineIsSkinned = false;
                                rp.setPipeline(this._shadowPipeline);
                                rp.setBindGroup(0, this._shadowVPBGs[cascadeSlot]!);

                                for (let i = 0; i < dynCount; i++) {
                                    const drawable = this._dynamicDrawables[i]!;
                                    const skinHandle = this._animationSystem?.getSkinForNode(drawable.nodeHandle);
                                    const isSkinned  = skinHandle !== undefined && skinHandle !== null;
                                    const drawData = this._meshSystem.getDrawData(drawable.meshHandle);
                                    if (!drawData) continue;

                                    if (isSkinned && !currentPipelineIsSkinned) {
                                        rp.setPipeline(this._shadowSkinnedPipeline!);
                                        rp.setBindGroup(0, this._shadowVPSkinnedBGs[cascadeSlot]!);
                                        currentPipelineIsSkinned = true;
                                    } else if (!isSkinned && currentPipelineIsSkinned) {
                                        rp.setPipeline(this._shadowPipeline);
                                        rp.setBindGroup(0, this._shadowVPBGs[cascadeSlot]!);
                                        currentPipelineIsSkinned = false;
                                    }

                                    if (isSkinned) {
                                        // Skinned draw: per-draw bind group with model matrix + joint buffer.
                                        const jointBuf = this._animationSystem!.getJointBuffer(skinHandle!);
                                        if (!jointBuf) continue;
                                        // Cache skinned shadow BGs: pool[jointBuf][drawIndex]
                                        let slotArr = this._skinnedShadowBGPool.get(jointBuf);
                                        if (!slotArr) { slotArr = []; this._skinnedShadowBGPool.set(jointBuf, slotArr); }
                                        let skinnedBG = slotArr[i];
                                        if (!skinnedBG) {
                                            const skinnedModelBGL = this._shadowSkinnedPipeline!.getBindGroupLayout(2);
                                            skinnedBG = this._backend.device.createBindGroup({
                                                label:   `Shadow/Skinned/G2/draw${i}`,
                                                layout:  skinnedModelBGL,
                                                entries: [
                                                    {
                                                        binding:  0,
                                                        resource: {
                                                            buffer: this._modelMatBuffer!,
                                                            offset: i * MODEL_UNIFORM_STRIDE,
                                                            size:   MODEL_MATRIX_SIZE,
                                                        },
                                                    },
                                                    { binding: 1, resource: { buffer: jointBuf } },
                                                ],
                                            });
                                            slotArr[i] = skinnedBG;
                                        }
                                        rp.setBindGroup(2, skinnedBG);
                                    } else {
                                        rp.setBindGroup(2, this._shadowModelMatBGs[i]!);
                                    }

                                    for (let j = 0; j < drawData.vertexBuffers.length; j++) {
                                        rp.setVertexBuffer(j, drawData.vertexBuffers[j]!);
                                    }
                                    const lod0 = drawData.lodLevels[0];
                                    if (!lod0) continue;
                                    for (const subMesh of lod0.subMeshes) {
                                        if (drawData.indexBuffer && subMesh.indexCount > 0) {
                                            rp.setIndexBuffer(drawData.indexBuffer, drawData.indexFormat);
                                            rp.drawIndexed(subMesh.indexCount, 1,
                                                subMesh.indexOffset, subMesh.vertexOffset);
                                        } else if (subMesh.vertexCount > 0) {
                                            rp.draw(subMesh.vertexCount, 1, subMesh.vertexOffset);
                                        }
                                    }
                                }
                            }

                            // ── Transparent shadow-casting objects ────────────
                            const transparents = this._cullResults.transparentDrawables;
                            if (transparents.length > 0) {
                                const opaqueCount = this._dynamicDrawables.length;
                                const maxTrans = Math.min(transparents.length, this._maxDraws - opaqueCount);
                                let stPipelineSet = false;

                                for (let ti = 0; ti < maxTrans; ti++) {
                                    const drawable = transparents[ti]!;
                                    const matRecord = this._materialSystem.getMaterial(drawable.materialHandle);
                                    if (!matRecord || !matRecord.castShadow) continue;

                                    if (!stPipelineSet) {
                                        // Filter to shadow-relevant defines only. The shadow shader
                                        // only supports HAS_TEXTURES, HAS_BASE_COLOR_MAP, DOUBLE_SIDED.
                                        const fullDefines = matRecord.shaderDefines;
                                        const defines: ShaderDefines = {};
                                        if (fullDefines['HAS_TEXTURES'])       defines['HAS_TEXTURES'] = '1';
                                        if (fullDefines['HAS_BASE_COLOR_MAP']) defines['HAS_BASE_COLOR_MAP'] = '1';
                                        if (fullDefines['DOUBLE_SIDED'])       defines['DOUBLE_SIDED'] = '1';
                                        const variantKey = 'st|' + _gbufferVariantKey(defines);
                                        let pipeline = this._shadowTransparentPipelineCache.get(variantKey);
                                        if (!pipeline) {
                                            const variant = this._shaderSystem.getVariant('shadow_transparent', defines);
                                            pipeline = this._pipelineManager.getOrCreateRenderPipeline({
                                                vertexShader:        variant.handle,
                                                fragmentShader:      variant.handle,
                                                vertexBufferLayouts: buildVertexBufferLayouts(STANDARD_VERTEX_LAYOUT),
                                                colorTargetFormats:  [],
                                                depthStencilFormat:  'depth32float',
                                                cullMode:            'back',
                                                depthWriteEnabled:   true,
                                                depthCompare:        'less',
                                            });
                                            this._shadowTransparentPipelineCache.set(variantKey, pipeline);
                                        }
                                        rp.setPipeline(pipeline);
                                        rp.setBindGroup(0, this._shadowVPTransparentBGs[cascadeSlot]!);
                                        stPipelineSet = true;
                                    }

                                    // @group(1): material bind group — created per-draw, NOT cached,
                                    // because the shadow pipeline's auto-layout BGL differs from
                                    // the forward pipeline's and they cannot share bind groups.
                                    // Use the active variant's BGL (the pipeline that was just set).
                                    const fullDef = matRecord.shaderDefines;
                                    const stDef: ShaderDefines = {};
                                    if (fullDef['HAS_TEXTURES'])       stDef['HAS_TEXTURES'] = '1';
                                    if (fullDef['HAS_BASE_COLOR_MAP']) stDef['HAS_BASE_COLOR_MAP'] = '1';
                                    if (fullDef['DOUBLE_SIDED'])       stDef['DOUBLE_SIDED'] = '1';
                                    const stKey = 'st|' + _gbufferVariantKey(stDef);
                                    const stActivePipeline = this._shadowTransparentPipelineCache.get(stKey);
                                    if (!stActivePipeline) continue;
                                    const stMatBGL = stActivePipeline.getBindGroupLayout(1);
                                    const stMatBG  = this._materialSystem.createShadowBindGroup(drawable.materialHandle, stMatBGL);
                                    if (!stMatBG) continue;
                                    rp.setBindGroup(1, stMatBG);

                                    // @group(2): model matrix
                                    const drawIdx = opaqueCount + ti;
                                    if (drawIdx >= this._shadowTransparentModelMatBGs.length) continue;
                                    rp.setBindGroup(2, this._shadowTransparentModelMatBGs[drawIdx]!);

                                    const drawData = this._meshSystem.getDrawData(drawable.meshHandle);
                                    if (!drawData) continue;
                                    for (let j = 0; j < drawData.vertexBuffers.length; j++) {
                                        rp.setVertexBuffer(j, drawData.vertexBuffers[j]!);
                                    }
                                    const lod0 = drawData.lodLevels[0];
                                    if (!lod0) continue;
                                    for (const subMesh of lod0.subMeshes) {
                                        if (drawData.indexBuffer && subMesh.indexCount > 0) {
                                            rp.setIndexBuffer(drawData.indexBuffer, drawData.indexFormat);
                                            rp.drawIndexed(subMesh.indexCount, 1,
                                                subMesh.indexOffset, subMesh.vertexOffset);
                                        } else if (subMesh.vertexCount > 0) {
                                            rp.draw(subMesh.vertexCount, 1, subMesh.vertexOffset);
                                        }
                                    }
                                }
                            }

                            cascadeSlot++;
                        }
                    }
                },
            );
        }

        // ---- Depth prepass --------------------------------------------------
        //
        // Renders all opaque geometry depth-only.  The G-Buffer pass below then
        // loads this depth buffer and uses depthCompare='less-equal', so the GPU
        // early-Z unit discards occluded fragments before the fragment shader runs.
        {
        const dpPipeline       = this._depthPrepassPipeline;
        const dpInstPipeline   = this._depthPrepassInstancedPipeline;
        const dpPerFrameGroups = this._depthPrepassPerFrameBindGroups;
        const dpInstPerFrameBGs = this._depthPrepassInstancedPerFrameBGs;
        const dpModelGroups    = this._depthPrepassModelMatBindGroups;
        const dpInstG2BG       = this._depthPrepassInstancedG2BG;

        this._renderGraph.addDepthPrepassPass(
            { resourceId: depth, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1.0 },
            (_encoder, passEncoder) => {
                if (!passEncoder || !dpPipeline) return;
                const rp   = passEncoder as GPURenderPassEncoder;
                const slot = this._frameIndex % this._backend.maxFramesInFlight;

                // ── Instanced static draws (one call per mesh+material batch) ──
                if (dpInstPipeline && dpInstG2BG && this._instanceBatches.length > 0) {
                    rp.setPipeline(dpInstPipeline);
                    rp.setBindGroup(0, dpInstPerFrameBGs[slot]!);
                    rp.setBindGroup(2, dpInstG2BG);

                    for (const batch of this._instanceBatches) {
                        const drawData = this._meshSystem.getDrawData(batch.meshHandle);
                        if (!drawData) continue;
                        for (let j = 0; j < drawData.vertexBuffers.length; j++) {
                            rp.setVertexBuffer(j, drawData.vertexBuffers[j]!);
                        }
                        const lod0 = drawData.lodLevels[0];
                        if (!lod0) continue;
                        for (const subMesh of lod0.subMeshes) {
                            if (drawData.indexBuffer && subMesh.indexCount > 0) {
                                rp.setIndexBuffer(drawData.indexBuffer, drawData.indexFormat);
                                rp.drawIndexed(subMesh.indexCount, batch.instanceCount,
                                    subMesh.indexOffset, subMesh.vertexOffset, batch.firstInstance);
                            } else if (subMesh.vertexCount > 0) {
                                rp.draw(subMesh.vertexCount, batch.instanceCount,
                                    subMesh.vertexOffset, batch.firstInstance);
                            }
                        }
                    }
                }

                // ── Per-draw dynamic objects ──────────────────────────────────
                //
                // Skinned drawables are skipped in the depth prepass because their
                // vertices move each frame and the prepass has no skinning pipeline.
                // The G-Buffer pass renders them with depthCompare='less-equal'
                // against the clear value (1.0), which always passes.
                const dynCount = Math.min(this._dynamicDrawables.length, this._maxDraws);
                if (dynCount > 0) {
                    rp.setPipeline(dpPipeline);
                    rp.setBindGroup(0, dpPerFrameGroups[slot]!);

                    for (let i = 0; i < dynCount; i++) {
                        const drawable = this._dynamicDrawables[i]!;
                        // Skip skinned meshes — depth prepass doesn't support skinning.
                        if (this._animationSystem?.getSkinForNode(drawable.nodeHandle) !== undefined) continue;
                        const drawData = this._meshSystem.getDrawData(drawable.meshHandle);
                        if (!drawData) continue;
                        rp.setBindGroup(2, dpModelGroups[i]!);
                        for (let j = 0; j < drawData.vertexBuffers.length; j++) {
                            rp.setVertexBuffer(j, drawData.vertexBuffers[j]!);
                        }
                        const lod0 = drawData.lodLevels[0];
                        if (!lod0) continue;
                        for (const subMesh of lod0.subMeshes) {
                            if (drawData.indexBuffer && subMesh.indexCount > 0) {
                                rp.setIndexBuffer(drawData.indexBuffer, drawData.indexFormat);
                                rp.drawIndexed(subMesh.indexCount, 1, subMesh.indexOffset, subMesh.vertexOffset);
                            } else if (subMesh.vertexCount > 0) {
                                rp.draw(subMesh.vertexCount, 1, subMesh.vertexOffset);
                            }
                        }
                    }
                }
            },
        );
        }

        // ---- Half-resolution depth downsample (placeholder) -----------------
        //
        // Declares a half-res r32float texture that downstream effects (SSAO,
        // SSR, Volumetric Fog, Contact Shadows) will read.  

        const halfW     = Math.max(1, Math.ceil(w / 2));
        const halfH     = Math.max(1, Math.ceil(h / 2));
        const HALF_DEPTH_USAGE = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
        const halfDepth = this._renderGraph.declareTexture('HalfDepth', {
            size:   [halfW, halfH],
            format: 'r32float',
            usage:  HALF_DEPTH_USAGE,
        });

        this._renderGraph.addPass({
            name:           'DepthDownsample',
            type:           PassType.Compute,
            reads:          [depth],
            writes:         [halfDepth],
            hasSideEffects: false,
            execute: (_encoder, passEncoder) => {
                const ddPipeline = this._depthDownsamplePipeline;
                if (!passEncoder || !ddPipeline) return;

                const srcView = this._resolveVirtualTexture(depth, { aspect: 'depth-only' });
                const dstView = this._resolveVirtualTexture(halfDepth);
                if (!srcView || !dstView) return;

                const bg = this._ddBGCache.getOrCreate(
                    [srcView, dstView],
                    () => this._backend.device.createBindGroup({
                        label:  'DepthDownsample/BG',
                        layout: ddPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: srcView },
                            { binding: 1, resource: dstView },
                        ],
                    }),
                );

                const cp = passEncoder as GPUComputePassEncoder;
                cp.setPipeline(ddPipeline);
                cp.setBindGroup(0, bg);
                cp.dispatchWorkgroups(
                    Math.ceil(halfW / 8),
                    Math.ceil(halfH / 8),
                );
            },
        });

        // ---- G-Buffer fill pass --------------------------------------------
        //
        // When the depth prepass ran, depth is loaded (the prepass already filled
        // it) and the pipeline uses depthCompare='less-equal' with depthBias=-2.
        // Without a prepass, depth is cleared here and the pipeline uses 'less'.

        const gbDepthLoadOp: GPULoadOp = 'load';

        this._renderGraph.addGBufferPass(
            [
                { resourceId: gbAlbedoAO,        loadOp: 'clear', storeOp: 'store', clearColor: { r: 0, g: 0, b: 0, a: 1 } },
                { resourceId: gbNormalRoughness,  loadOp: 'clear', storeOp: 'store', clearColor: { r: 0.5, g: 0.5, b: 1, a: 0 } },
                { resourceId: gbMetallicEmissive, loadOp: 'clear', storeOp: 'store', clearColor: { r: 0, g: 0, b: 0, a: 1 } },
            ],
            { resourceId: depth, depthLoadOp: gbDepthLoadOp, depthStoreOp: 'store', depthClearValue: 1.0 },
            (_encoder, passEncoder) => {
                if (!passEncoder) return;
                // addGBufferPass always opens a GPURenderPassEncoder — cast from the union type.
                const rp   = passEncoder as GPURenderPassEncoder;
                const slot = this._frameIndex % this._backend.maxFramesInFlight;

                // null sentinel — guaranteed to differ from any real key on the first draw.
                let activePipelineKey: string | null = null;
                let activePerFrameBGs: GPUBindGroup[] = [];
                let activeModelMatBGs: GPUBindGroup[] = [];

                // ── Instanced static batches (one draw call per mesh+material) ──
                //
                // Each batch uses the USE_INSTANCING variant of the GBuffer pipeline.
                // The storage buffer holding all instance world matrices is bound
                // once to @group(2); firstInstance in drawIndexed() selects the
                // starting entry for each batch.
                for (const batch of this._instanceBatches) {
                    const drawData = this._meshSystem.getDrawData(batch.meshHandle);
                    if (!drawData) continue;

                    const instancedDefines = { ...batch.baseDefines, USE_INSTANCING: '1' };
                    // Full cache key includes the 'dp|' prefix when the prepass is active.
                    const variantKey = ('dp|') + _gbufferVariantKey(instancedDefines);

                    if (variantKey !== activePipelineKey) {
                        const pipeline = this._getOrCreateGBufferVariant(instancedDefines);
                        rp.setPipeline(pipeline);
                        activePipelineKey = variantKey;
                        activePerFrameBGs = this._gbufferPerFrameBGCache.get(variantKey)!;
                        // @group(2) for instanced variants is a single storage-buffer BG.
                        activeModelMatBGs = []; // not used in instanced path
                        rp.setBindGroup(0, activePerFrameBGs[slot]!);
                    }

                    const activePipeline = this._gbufferPipelineCache.get(activePipelineKey!)!;
                    const matBGL = activePipeline.getBindGroupLayout(1);
                    const matBG  = this._materialSystem.getOrCreateBindGroup(batch.materialHandle, matBGL);
                    if (!matBG) continue;

                    const instG2BG = this._instancedG2BGCache.get(activePipelineKey!);
                    if (!instG2BG) continue;

                    rp.setBindGroup(1, matBG);
                    rp.setBindGroup(2, instG2BG);

                    for (let j = 0; j < drawData.vertexBuffers.length; j++) {
                        rp.setVertexBuffer(j, drawData.vertexBuffers[j]!);
                    }

                    const lod0 = drawData.lodLevels[0];
                    if (!lod0) continue;

                    for (const subMesh of lod0.subMeshes) {
                        if (drawData.indexBuffer && subMesh.indexCount > 0) {
                            rp.setIndexBuffer(drawData.indexBuffer, drawData.indexFormat);
                            rp.drawIndexed(subMesh.indexCount, batch.instanceCount,
                                subMesh.indexOffset, subMesh.vertexOffset, batch.firstInstance);
                            this._stats.drawCalls++;
                            this._stats.triangles += ((subMesh.indexCount / 3) | 0) * batch.instanceCount;
                        } else if (subMesh.vertexCount > 0) {
                            rp.draw(subMesh.vertexCount, batch.instanceCount,
                                subMesh.vertexOffset, batch.firstInstance);
                            this._stats.drawCalls++;
                            this._stats.triangles += ((subMesh.vertexCount / 3) | 0) * batch.instanceCount;
                        }
                    }
                }

                // ── Per-draw dynamic objects ──────────────────────────────────
                //
                // Dynamic objects (isStatic=false) use the classic per-draw
                // uniform path: each draw reads its model matrix from its own
                // 256-byte slot in the model-matrix ring buffer.
                //
                // Skinned objects additionally bind a joint matrix storage buffer
                // at @group(2) @binding(1) and use the USE_SKINNING pipeline variant.
                const dynCount = Math.min(this._dynamicDrawables.length, this._maxDraws);
                for (let i = 0; i < dynCount; i++) {
                    const drawable = this._dynamicDrawables[i]!;
                    const drawData = this._meshSystem.getDrawData(drawable.meshHandle);
                    if (!drawData) continue;

                    // Check if this drawable is skinned (has a registered skin in the animation system).
                    const skinHandle = this._animationSystem?.getSkinForNode(drawable.nodeHandle);
                    const isSkinned  = skinHandle !== undefined && skinHandle !== null;

                    // Select the pipeline variant matching this material's defines.
                    // Auto-layout BGLs are pipeline-opaque, so groups 0, 1, and 2
                    // must all come from the ACTIVE pipeline's BGLs.
                    const matRecord  = this._materialSystem.getMaterial(drawable.materialHandle);
                    const defines    = { ...matRecord?.shaderDefines };
                    if (isSkinned) defines['USE_SKINNING'] = '1';
                    // Full cache key includes the 'dp|' prefix when the prepass is active.
                    const variantKey = ('dp|') + _gbufferVariantKey(defines);

                    if (variantKey !== activePipelineKey) {
                        const pipeline = this._getOrCreateGBufferVariant(defines);
                        rp.setPipeline(pipeline);
                        activePipelineKey = variantKey;
                        activePerFrameBGs = this._gbufferPerFrameBGCache.get(activePipelineKey)!;
                        activeModelMatBGs = this._gbufferModelMatBGCache.get(activePipelineKey)!;
                        rp.setBindGroup(0, activePerFrameBGs[slot]!);
                    }

                    const activePipeline = this._gbufferPipelineCache.get(activePipelineKey!)!;
                    const matBGL = activePipeline.getBindGroupLayout(1);
                    const matBG  = this._materialSystem.getOrCreateBindGroup(drawable.materialHandle, matBGL);
                    if (!matBG) continue;

                    rp.setBindGroup(1, matBG);

                    if (isSkinned) {
                        // Skinned draw: per-draw bind group with model matrix + joint buffer.
                        const jointBuf = this._animationSystem!.getJointBuffer(skinHandle!);
                        if (!jointBuf) continue;
                        // Cache skinned GBuffer BGs: pool[variantKey][jointBuf][drawIndex]
                        let variantPool = this._skinnedGBufBGPool.get(activePipelineKey!);
                        if (!variantPool) { variantPool = new Map(); this._skinnedGBufBGPool.set(activePipelineKey!, variantPool); }
                        let slotArr = variantPool.get(jointBuf);
                        if (!slotArr) { slotArr = []; variantPool.set(jointBuf, slotArr); }
                        let skinnedBG = slotArr[i];
                        if (!skinnedBG) {
                            const mmBGL = activePipeline.getBindGroupLayout(2);
                            skinnedBG = this._backend.device.createBindGroup({
                                label:   `GBuffer/Skinned/G2/draw${i}`,
                                layout:  mmBGL,
                                entries: [
                                    {
                                        binding:  0,
                                        resource: {
                                            buffer: this._modelMatBuffer!,
                                            offset: i * MODEL_UNIFORM_STRIDE,
                                            size:   MODEL_MATRIX_SIZE,
                                        },
                                    },
                                    { binding: 1, resource: { buffer: jointBuf } },
                                ],
                            });
                            slotArr[i] = skinnedBG;
                        }
                        rp.setBindGroup(2, skinnedBG);
                    } else {
                        rp.setBindGroup(2, activeModelMatBGs[i]!);
                    }

                    for (let j = 0; j < drawData.vertexBuffers.length; j++) {
                        rp.setVertexBuffer(j, drawData.vertexBuffers[j]!);
                    }

                    const lod0 = drawData.lodLevels[0];
                    if (!lod0) continue;

                    for (const subMesh of lod0.subMeshes) {
                        if (drawData.indexBuffer && subMesh.indexCount > 0) {
                            rp.setIndexBuffer(drawData.indexBuffer, drawData.indexFormat);
                            rp.drawIndexed(subMesh.indexCount, 1, subMesh.indexOffset, subMesh.vertexOffset);
                            this._stats.drawCalls++;
                            this._stats.triangles += (subMesh.indexCount / 3) | 0;
                        } else if (subMesh.vertexCount > 0) {
                            rp.draw(subMesh.vertexCount, 1, subMesh.vertexOffset);
                            this._stats.drawCalls++;
                            this._stats.triangles += (subMesh.vertexCount / 3) | 0;
                        }
                    }
                }
            },
        );

        // ---- Deferred lighting pass ----------------------------------------

        this._renderGraph.addLightingPass(
            [gbAlbedoAO, gbNormalRoughness, gbMetallicEmissive, depth],
            [{ resourceId: hdrColor, loadOp: 'clear', storeOp: 'store', clearColor: { r: 0, g: 0, b: 0, a: 1 } }],
            (_encoder, passEncoder) => {
                const lp        = this._lightingPipeline;
                const gbBGL     = this._lightingGBufferBGL;
                const lightsBG  = this._lightingLightsBG;
                const sampler   = this._clampSampler;
                const slot      = this._frameIndex % this._backend.maxFramesInFlight;
                const lpFrameBG = this._lightingPerFrameBindGroups[slot];

                if (!passEncoder || !lp || !gbBGL || !lightsBG || !sampler || !lpFrameBG) return;

                // Resolve G-Buffer virtual ids to views (only valid after compile()).
                const albedoView   = this._resolveVirtualTexture(gbAlbedoAO);
                const normalView   = this._resolveVirtualTexture(gbNormalRoughness);
                const metallicView = this._resolveVirtualTexture(gbMetallicEmissive);
                const depthView    = this._resolveVirtualTexture(depth, { aspect: 'depth-only' });
                if (!albedoView || !normalView || !metallicView || !depthView) return;

                // Cache G-Buffer bind group — invalidated when views change (pool recycling).
                const gbufferBG = this._lightingGBufBGCache.getOrCreate(
                    [albedoView, normalView, metallicView, depthView],
                    () => this._backend.device.createBindGroup({
                        label:   'Lighting/GBuffer/BG',
                        layout:  gbBGL!,
                        entries: [
                            { binding: 0, resource: albedoView! },
                            { binding: 1, resource: normalView! },
                            { binding: 2, resource: metallicView! },
                            { binding: 3, resource: depthView! },
                            { binding: 4, resource: sampler! },
                        ],
                    }),
                );

                const rp = passEncoder as GPURenderPassEncoder;
                rp.setPipeline(lp);
                rp.setBindGroup(0, lpFrameBG);
                rp.setBindGroup(1, gbufferBG);
                rp.setBindGroup(2, lightsBG);
                if (this._lightingEnvBG) rp.setBindGroup(3, this._lightingEnvBG);
                rp.draw(3);   // fullscreen triangle — no vertex buffer needed
            },
        );

        // ---- Background pass (after lighting, before forward transparent) ----
        //
        // Renders the scene background into hdrColor where depth == 1.0 (sky
        // pixels not covered by geometry).  Uses depthCompare='equal' so only
        // far-plane fragments pass.

        if (this._backgroundSystem && this._placeholder2DView) {
            const bgSystem    = this._backgroundSystem;
            const invVPRef    = this._lastInvViewProj;
            const placeholder = this._placeholder2DView;

            this._renderGraph.addPass({
                name:  'Background',
                type:  PassType.Render,
                reads: [depth],
                writes: [],
                hasSideEffects: true,
                colorAttachments: [{ resourceId: hdrColor, loadOp: 'load', storeOp: 'store' }],
                depthAttachment:  { resourceId: depth, depthLoadOp: 'load', depthStoreOp: 'discard', depthReadOnly: true },
                execute: (_encoder, passEncoder) => {
                    if (!passEncoder) return;
                    bgSystem.render(passEncoder as GPURenderPassEncoder, invVPRef, placeholder);
                },
            });
        }

        // ---- Post-process chain --------------------------------------------

        const ppCtx = this._makePostProcessContext(gbAlbedoAO, gbNormalRoughness, gbMetallicEmissive);
        let finalColor = this._postProcessStack.addPasses(this._renderGraph, ppCtx, hdrColor, depth, halfDepth);

        // ---- Forward transparent pass (after post-process) -----------------
        //
        // Transparent materials are rendered in a forward PBR pass.
        // A copy of the scene color is taken first so the shader can sample the
        // background at refracted UV coordinates (screen-space refraction).
        // Sorted back-to-front by the culler for correct painter's algorithm.

        if (this._cullResults.transparentDrawables.length > 0) {
            // forwardOutput: render target for transparent objects.
            // Initialized via a fullscreen blit from finalColor (the post-processed
            // scene), then transparent objects are drawn on top.
            const forwardOutput = this._renderGraph.declareTexture('ForwardOutput', {
                size:   [w, h],
                format: 'rgba16float',
                usage:  COLOR,
            });

            // Half-res refraction source: downsample finalColor to half resolution.
            // Transparent objects sample this for screen-space refraction — half-res
            // is sufficient because refraction is a low-frequency effect (smooth
            // surfaces, UV offset already blurs detail) and improves texture cache
            // coherency during the forward pass.
            const halfRefractionColor = this._renderGraph.declareTexture('HalfRefractionColor', {
                size:   [halfW, halfH],
                format: 'rgba16float',
                usage:  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            });

            // Blit pass: draw finalColor into forwardOutput as a fullscreen quad.
            // This avoids copyTextureToTexture (which requires COPY_SRC on the
            // source, but post-process output textures don't have it).
            const blitSrcId = finalColor;
            const blitPipeline = this._fullscreenCopyPipeline;
            const blitSampler  = this._clampSampler;
            if (blitPipeline && blitSampler) {
                // Blit full-res finalColor → forwardOutput (composite target).
                this._renderGraph.addPass({
                    name:           'ForwardSceneBlit',
                    type:           PassType.Render,
                    reads:          [blitSrcId],
                    colorAttachments: [{ resourceId: forwardOutput, loadOp: 'clear', storeOp: 'store', clearColor: { r: 0, g: 0, b: 0, a: 1 } }],
                    colorFormats:   ['rgba16float'],
                    hasSideEffects: false,
                    execute: (_encoder, passEncoder) => {
                        if (!passEncoder) return;
                        const srcView = this._resolveVirtualTexture(blitSrcId);
                        if (!srcView) return;
                        const bg = this._fwdSceneBlitBGCache.getOrCreate(
                            [srcView],
                            () => this._backend.device.createBindGroup({
                                label:  'ForwardSceneBlit/BG',
                                layout: blitPipeline.getBindGroupLayout(0),
                                entries: [
                                    { binding: 0, resource: srcView },
                                    { binding: 1, resource: blitSampler },
                                ],
                            }),
                        );
                        const rp = passEncoder as GPURenderPassEncoder;
                        rp.setPipeline(blitPipeline);
                        rp.setBindGroup(0, bg);
                        rp.draw(3);
                    },
                });

                // Blit full-res finalColor → half-res refraction source.
                // Reuses the same fullscreen copy pipeline (rgba16float target).
                // The render pass viewport is set by the half-res attachment;
                // the shader samples the full-res source via normalized UVs,
                // so bilinear filtering naturally downscales.
                this._renderGraph.addPass({
                    name:           'RefractionDownsample',
                    type:           PassType.Render,
                    reads:          [blitSrcId],
                    colorAttachments: [{ resourceId: halfRefractionColor, loadOp: 'clear', storeOp: 'store', clearColor: { r: 0, g: 0, b: 0, a: 1 } }],
                    colorFormats:   ['rgba16float'],
                    hasSideEffects: false,
                    execute: (_encoder, passEncoder) => {
                        if (!passEncoder) return;
                        const srcView = this._resolveVirtualTexture(blitSrcId);
                        if (!srcView) return;
                        const bg = this._refrDsBGCache.getOrCreate(
                            [srcView],
                            () => this._backend.device.createBindGroup({
                                label:  'RefractionDownsample/BG',
                                layout: blitPipeline.getBindGroupLayout(0),
                                entries: [
                                    { binding: 0, resource: srcView },
                                    { binding: 1, resource: blitSampler },
                                ],
                            }),
                        );
                        const rp = passEncoder as GPURenderPassEncoder;
                        rp.setPipeline(blitPipeline);
                        rp.setBindGroup(0, bg);
                        rp.draw(3);
                    },
                });
            }

            // Forward pass: draw transparent objects into forwardOutput while
            // sampling halfRefractionColor for refraction.
            // No cycle: reads halfRefractionColor, writes forwardOutput (distinct resources).
            const refractionSrcId = halfRefractionColor;
            this._renderGraph.addForwardPass(
                [depth, refractionSrcId],
                [{ resourceId: forwardOutput, loadOp: 'load', storeOp: 'store' }],
                { resourceId: depth, depthLoadOp: 'load', depthStoreOp: 'discard', depthReadOnly: true },
                (_encoder, passEncoder) => {
                    if (!passEncoder) return;
                    const fwdLightEnvBG = this._forwardLightEnvBG;
                    if (!fwdLightEnvBG) return;

                    const rp   = passEncoder as GPURenderPassEncoder;
                    const slot = this._frameIndex % this._backend.maxFramesInFlight;

                    // Resolve scene color for @group(0) refraction sampling
                    const sceneColorView = this._resolveVirtualTexture(refractionSrcId);
                    if (!sceneColorView) return;

                    const linearClampSampler = this._forwardLinearSampler!;

                    const transparents = this._cullResults.transparentDrawables;
                    const opaqueCount = this._dynamicDrawables.length;
                    const maxTransparent = Math.min(transparents.length, this._maxDraws - opaqueCount);
                    if (maxTransparent <= 0) return;

                    // Transparent world matrices already uploaded by _packTransparentMatrices().
                    let activePipelineKey: string | null = null;

                    for (let i = 0; i < maxTransparent; i++) {
                        const drawable = transparents[i]!;
                        const matRecord = this._materialSystem.getMaterial(drawable.materialHandle);
                        const defines = matRecord?.shaderDefines ?? {};
                        const variantKey = _gbufferVariantKey(defines);

                        if (variantKey !== activePipelineKey) {
                            let pipeline = this._forwardPipelineCache.get(variantKey);
                            if (!pipeline) {
                                const variant = this._shaderSystem.getVariant('forward_transparent', defines);
                                pipeline = this._pipelineManager.getOrCreateForwardPipeline(
                                    variant.handle, variant.handle, STANDARD_VERTEX_LAYOUT,
                                    {
                                        blend: BLEND_STRAIGHT_ALPHA,
                                        doubleSided: defines['DOUBLE_SIDED'] === '1',
                                    },
                                );
                                this._forwardPipelineCache.set(variantKey, pipeline);
                            }
                            rp.setPipeline(pipeline);
                            activePipelineKey = variantKey;

                            // @group(0): per-frame uniforms + scene color for refraction
                            let fwdBGC = this._fwdPerFrameBGCache.get(variantKey);
                            if (!fwdBGC) { fwdBGC = new BindGroupCache(); this._fwdPerFrameBGCache.set(variantKey, fwdBGC); }
                            const g0BGL = pipeline.getBindGroupLayout(0);
                            const g0BG = fwdBGC.getOrCreate(
                                [sceneColorView, this._perFrameBuffers[slot]!],
                                () => this._backend.device.createBindGroup({
                                    label:   'Forward/PerFrame/BG',
                                    layout:  g0BGL,
                                    entries: [
                                        { binding: 0, resource: { buffer: this._perFrameBuffers[slot]! } },
                                        { binding: 1, resource: sceneColorView! },
                                        { binding: 2, resource: linearClampSampler },
                                    ],
                                }),
                            );
                            rp.setBindGroup(0, g0BG);
                            rp.setBindGroup(3, fwdLightEnvBG);
                        }

                        // @group(1): material
                        const activePipeline = this._forwardPipelineCache.get(activePipelineKey!)!;
                        const matBGL = activePipeline.getBindGroupLayout(1);
                        const matBG  = this._materialSystem.getOrCreateBindGroup(drawable.materialHandle, matBGL);
                        if (!matBG) continue;
                        rp.setBindGroup(1, matBG);

                        // @group(2): model matrix
                        const drawIdx = opaqueCount + i;
                        if (drawIdx < this._forwardModelMatBGs.length) {
                            rp.setBindGroup(2, this._forwardModelMatBGs[drawIdx]!);
                        } else {
                            continue;
                        }

                        // Draw mesh
                        const drawData = this._meshSystem.getDrawData(drawable.meshHandle);
                        if (!drawData) continue;
                        for (let j = 0; j < drawData.vertexBuffers.length; j++) {
                            rp.setVertexBuffer(j, drawData.vertexBuffers[j]!);
                        }
                        const lod0 = drawData.lodLevels[0];
                        if (!lod0) continue;
                        for (const subMesh of lod0.subMeshes) {
                            if (drawData.indexBuffer && subMesh.indexCount > 0) {
                                rp.setIndexBuffer(drawData.indexBuffer, drawData.indexFormat);
                                rp.drawIndexed(subMesh.indexCount, 1, subMesh.indexOffset, subMesh.vertexOffset);
                                this._stats.drawCalls++;
                                this._stats.triangles += (subMesh.indexCount / 3) | 0;
                            } else if (subMesh.vertexCount > 0) {
                                rp.draw(subMesh.vertexCount, 1, subMesh.vertexOffset);
                                this._stats.drawCalls++;
                                this._stats.triangles += (subMesh.vertexCount / 3) | 0;
                            }
                        }
                    }
                },
            );

            // Forward output replaces finalColor for the blit pass.
            finalColor = forwardOutput;
        }

        // ---- Present (blit HDR → swap-chain with tonemapping) ---------------
        // Capture finalColor id for the closure.
        const finalColorId = finalColor;

        this._renderGraph.addPass({
            name:           'Present',
            type:           PassType.Render,
            reads:          [finalColorId],
            hasSideEffects: true,
            execute: (encoder, _passEncoder) => {
                const presentPass = encoder.beginRenderPass({
                    label: 'Present',
                    colorAttachments: [{
                        view:       this._backend.getCurrentTextureView(),
                        loadOp:     'clear',
                        storeOp:    'store',
                        clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    }],
                });

                const bp      = this._blitPipeline;
                const bbl     = this._blitBGL;
                const sampler = this._clampSampler;
                if (bp && bbl && sampler) {
                    const finalView = this._resolveVirtualTexture(finalColorId);
                    if (finalView) {
                        const blitBG = this._blitPresentBGCache.getOrCreate(
                            [finalView],
                            () => this._backend.device.createBindGroup({
                                label:   'Blit/BG',
                                layout:  bbl!,
                                entries: [
                                    { binding: 0, resource: finalView },
                                    { binding: 1, resource: sampler! },
                                    { binding: 2, resource: { buffer: this._blitExposureBuffer! } },
                                ],
                            }),
                        );
                        presentPass.setPipeline(bp);
                        presentPass.setBindGroup(0, blitBG);
                        presentPass.draw(3);
                    }
                }

                presentPass.end();
            },
        });
    }

    private _compileAndExecute(): void {
        const compiled = this._renderGraph.compile();
        this._stats.passCount = compiled.orderedPasses.length;
        this._renderGraph.execute();

    }

    private _endFrame(cpuTimeMs: number): void {
        this._stats.frameIndex = this._frameIndex;
        this._stats.cpuTimeMs  = cpuTimeMs;
    }

    // -------------------------------------------------------------------------
    // Per-frame uniform upload
    // -------------------------------------------------------------------------

    private _uploadPerFrameUniforms(uniforms: PerFrameUniforms): void {
        const slot   = this._frameIndex % this._backend.maxFramesInFlight;
        const buffer = this._perFrameBuffers[slot];
        if (!buffer) return;

        const data = this._perFrameData;
        let off = 0;

        data.set(uniforms.viewMatrix,            off); off += 16;
        data.set(uniforms.projectionMatrix,      off); off += 16;
        data.set(uniforms.viewProjectionMatrix,  off); off += 16;
        data.set(uniforms.inverseViewProjection, off); off += 16;
        // off = 64 = 256 bytes

        data.set(uniforms.cameraPosition, off); off += 3;
        off++; // _pad0
        // off = 68 = 272 bytes

        data[off++] = uniforms.time;
        data[off++] = uniforms.deltaTime;
        data[off++] = uniforms.resolution[0];
        data[off++] = uniforms.resolution[1];
        // off = 72 = 288 bytes

        data[off++] = uniforms.frameIndex;
        data[off++] = uniforms.exposure;
        data[off++] = uniforms.jitter[0];
        data[off++] = uniforms.jitter[1];
        // off = 76 = 304 bytes ✓

        this._backend.queue.writeBuffer(buffer, 0, data);
    }

    // -------------------------------------------------------------------------
    // Public accessors for render callbacks
    // -------------------------------------------------------------------------

    /** GPUBindGroup for @group(0): the per-frame uniform buffer (current ring slot). */
    getPerFrameBindGroup(): GPUBindGroup | undefined {
        const slot = this._frameIndex % this._backend.maxFramesInFlight;
        return this._perFrameBindGroups[slot];
    }

    /** GPUBuffer for the current ring slot (for manual bind group creation). */
    getPerFrameBuffer(): GPUBuffer | undefined {
        const slot = this._frameIndex % this._backend.maxFramesInFlight;
        return this._perFrameBuffers[slot];
    }

    /** GPUBindGroupLayout for @group(0) — use when building material/mesh bind groups. */
    getPerFrameBindGroupLayout(): GPUBindGroupLayout {
        return this._perFrameBindGroupLayout;
    }

    /** Cull results from the current frame (populated after _updateScene). */
    getCullResults(): Readonly<CullResults> {
        return this._cullResults;
    }

    // -------------------------------------------------------------------------
    // Environment cubemap (IBL reflections)
    // -------------------------------------------------------------------------

    /**
     * Set the environment cubemap used for IBL specular reflections.
     * Called by SceneLoader after loading an envConfig cubemap.
     *
     * TODO: When reflection probes are implemented, this method should be
     *       extended (or replaced) to register per-probe cubemaps. The
     *       lighting pass would then select the closest probe per pixel
     *       instead of the single global cubemap.
     *
     * @param cubemapHandle ResourceHandle of a 6-layer texture (cube).
     */
    setEnvironmentCubemap(cubemapHandle: ResourceHandle): void {
        if (!this._lightingEnvBGL || !this._envParamsBuffer || !this._envCubeSampler) return;

        const cubeTex = this._resources.getTexture(cubemapHandle);
        if (!cubeTex) return;

        this._envCubemapTexture = cubeTex;

        // Enable IBL in the uniform (preserves current ambient color)
        this._uploadEnvParams(1);

        // Rebuild deferred + forward bind groups with the actual cubemap
        const cubeView = cubeTex.createView({ dimension: 'cube', arrayLayerCount: 6 });
        this._lightingEnvBG = this._backend.device.createBindGroup({
            label:  'Lighting/Env/BG',
            layout: this._lightingEnvBGL,
            entries: [
                { binding: 0, resource: { buffer: this._envParamsBuffer } },
                { binding: 1, resource: cubeView },
                { binding: 2, resource: this._envCubeSampler },
            ],
        });
        this._rebuildForwardLightEnvBG(cubeTex);
    }

    /**
     * Remove the environment cubemap, restoring the placeholder.
     * Disables IBL reflections in the deferred and forward lighting passes.
     */
    clearEnvironmentCubemap(): void {
        if (!this._lightingEnvBGL || !this._envParamsBuffer || !this._envCubeSampler || !this._placeholderCubeTex) return;

        this._envCubemapTexture = null;
        this._uploadEnvParams(0);

        this._lightingEnvBG = this._backend.device.createBindGroup({
            label:  'Lighting/Env/BG',
            layout: this._lightingEnvBGL,
            entries: [
                { binding: 0, resource: { buffer: this._envParamsBuffer } },
                { binding: 1, resource: this._placeholderCubeTex.createView({ dimension: 'cube' }) },
                { binding: 2, resource: this._envCubeSampler },
            ],
        });
        this._rebuildForwardLightEnvBG(this._placeholderCubeTex);
    }

    /** Rebuild the forward transparent pass @group(3) with the given cube texture. */
    private _rebuildForwardLightEnvBG(cubeTex: GPUTexture): void {
        if (!this._forwardLightEnvBGL || !this._envParamsBuffer || !this._envCubeSampler) return;

        const lightGpuBuffer   = this._resources.getBuffer(this._lightSystem.getLightStorageBuffer());
        const shadowAtlasGPU   = this._resources.getTexture(this._lightSystem.getShadowAtlasTexture());
        const shadowDataGpuBuf = this._resources.getBuffer(this._lightSystem.getShadowDataBuffer());
        if (!lightGpuBuffer || !shadowAtlasGPU || !shadowDataGpuBuf) return;

        this._forwardLightEnvBG = this._backend.device.createBindGroup({
            label:  'Forward/LightEnv/BG',
            layout: this._forwardLightEnvBGL,
            entries: [
                { binding: 0, resource: { buffer: lightGpuBuffer } },
                { binding: 1, resource: shadowAtlasGPU.createView({ aspect: 'depth-only' }) },
                { binding: 2, resource: this._shadowCompSampler! },
                { binding: 3, resource: { buffer: shadowDataGpuBuf } },
                { binding: 4, resource: { buffer: this._envParamsBuffer } },
                { binding: 5, resource: cubeTex.createView({ dimension: 'cube', arrayLayerCount: 6 }) },
                { binding: 6, resource: this._envCubeSampler },
            ],
        });
    }

    /**
     * Set the scene ambient light color (linear RGB).
     * Replaces the previous hardcoded 0.03 gray.
     */
    setAmbientColor(r: number, g: number, b: number): void {
        this._ambientColor = [r, g, b];
        // Re-upload with the current cubemap-enabled flag preserved
        if (this._envParamsBuffer) {
            // Read back the enabled flag from the cached ambient state isn't
            // necessary — we can peek at whether the bind group uses the real
            // cubemap.  Simpler: just read the first u32 from the buffer isn't
            // possible synchronously, so track it.
            this._uploadEnvParams(this._envCubemapEnabled ? 1 : 0);
        }
    }

    /** Whether the env cubemap is currently active (for re-upload). */
    private _envCubemapEnabled = false;
    /** Cached env cubemap GPU texture for post-process effects (SSR fallback). */
    private _envCubemapTexture: GPUTexture | null = null;

    /** Write the 16-byte EnvParams uniform (enabled u32 + ambient rgb f32×3). */
    private _uploadEnvParams(enabled: number): void {
        if (!this._envParamsBuffer) return;
        this._envCubemapEnabled = enabled !== 0;
        const data = new ArrayBuffer(16);
        new Uint32Array(data, 0, 1)[0] = enabled;
        const floats = new Float32Array(data, 4, 3);
        floats[0] = this._ambientColor[0];
        floats[1] = this._ambientColor[1];
        floats[2] = this._ambientColor[2];
        this._backend.queue.writeBuffer(this._envParamsBuffer, 0, data);
    }

    // -------------------------------------------------------------------------
    // G-Buffer pipeline variant cache
    // -------------------------------------------------------------------------

    /**
     * Return (or compile and cache) the GBuffer pipeline variant for the given
     * set of shader defines.  Each unique define combination produces a separate
     * GPURenderPipeline with its own auto-derived bind-group layouts.
     */
    private _getOrCreateGBufferVariant(defines: ShaderDefines): GPURenderPipeline {
        // Include the depth-prepass flag in the cache key so variants compiled
        // with prepass=true (depthCompare 'less-equal', depthBias -2) and
        // prepass=false (depthCompare 'less', depthBias 0) are stored separately.
        const key = ('dp|') + _gbufferVariantKey(defines);
        let pipeline = this._gbufferPipelineCache.get(key);
        if (!pipeline) {
            const isSkinned = defines['USE_SKINNING'] === '1';
            const vertexLayout = isSkinned ? SKINNED_VERTEX_LAYOUT : STANDARD_VERTEX_LAYOUT;
            const variant = this._shaderSystem.getVariant('gbuffer', defines);
            // Skinned meshes are skipped in the depth prepass (no skinning
            // pipeline there), so their G-Buffer variant must write depth
            // itself — use depthPrepass=false to get depthWriteEnabled=true
            // and depthCompare='less'.
            pipeline = this._pipelineManager.getOrCreateGBufferPipeline(
                variant.handle,
                variant.handle,
                vertexLayout,
                {
                    depthPrepass: !isSkinned,
                    doubleSided:  defines['DOUBLE_SIDED'] === '1',
                    alphaMask:    defines['ALPHA_MASK']   === '1',
                },
            );
            this._gbufferPipelineCache.set(key, pipeline);

            // Auto-layout BGLs are pipeline-opaque: a bind group created from
            // variant A's BGL is NOT compatible with variant B, even if their
            // WGSL declarations are identical.  Create per-variant bind groups
            // for @group(0) (per-frame) and @group(2) (model matrix) so each
            // variant has fully compatible bind groups.

            // @group(0) — one bind group per ring slot.
            const pfBGL  = pipeline.getBindGroupLayout(0);
            const pfBGs: GPUBindGroup[] = [];
            for (let i = 0; i < this._backend.maxFramesInFlight; i++) {
                pfBGs.push(this._backend.device.createBindGroup({
                    label:   `GBuffer[${key || 'base'}]/PerFrame/BG[${i}]`,
                    layout:  pfBGL,
                    entries: [{ binding: 0, resource: { buffer: this._perFrameBuffers[i]! } }],
                }));
            }
            this._gbufferPerFrameBGCache.set(key, pfBGs);

            // @group(2) — behaviour depends on the pipeline variant:
            //
            //   USE_INSTANCING=1  →  single storage-buffer BG covering the
            //     entire _staticInstanceBuffer; cached in _instancedG2BGCache.
            //     The per-draw model-matrix BG array is left empty (unused).
            //
            //   (default)         →  one uniform sub-range BG per MAX_DRAWS slot;
            //     cached in _gbufferModelMatBGCache.
            const mmBGL = pipeline.getBindGroupLayout(2);
            const isInstanced = defines['USE_INSTANCING'] === '1';

            if (isInstanced) {
                const instBG = this._backend.device.createBindGroup({
                    label:   `GBuffer[${key}]/InstancedG2/BG`,
                    layout:  mmBGL,
                    entries: [{ binding: 0, resource: { buffer: this._staticInstanceBuffer! } }],
                });
                this._instancedG2BGCache.set(key, instBG);
                this._gbufferModelMatBGCache.set(key, []); // unused for instanced variants
            } else if (isSkinned) {
                // Skinned variants have @group(2) = { binding(0): model matrix, binding(1): joint buffer }.
                // The joint buffer varies per-skin, so bind groups are created per-draw in the render loop.
                this._gbufferModelMatBGCache.set(key, []); // unused — created lazily per draw
            } else {
                const mmBGs: GPUBindGroup[] = [];
                for (let i = 0; i < this._maxDraws; i++) {
                    mmBGs.push(this._backend.device.createBindGroup({
                        label:   `GBuffer[${key || 'base'}]/ModelMatrix/BG[${i}]`,
                        layout:  mmBGL,
                        entries: [{
                            binding:  0,
                            resource: {
                                buffer: this._modelMatBuffer!,
                                offset: i * MODEL_UNIFORM_STRIDE,
                                size:   MODEL_MATRIX_SIZE,
                            },
                        }],
                    }));
                }
                this._gbufferModelMatBGCache.set(key, mmBGs);
            }
        }
        return pipeline;
    }

    /**
     * Resolve a virtual resource id (from the current frame's render graph)
     * to a GPUTextureView.  Only valid inside execute() callbacks — i.e. after
     * compile() has allocated physical handles.
     */
    private _resolveVirtualTexture(
        id: number,
        viewDesc?: GPUTextureViewDescriptor,
    ): GPUTextureView | undefined {
        const handle = this._renderGraph.getPhysicalHandle(id);
        if (handle === undefined) return undefined;
        return this._resources.createView(handle, viewDesc);
    }

    // -------------------------------------------------------------------------
    // Post-process context
    // -------------------------------------------------------------------------

    /**
     * Build a PostProcessContext suitable for init().
     * G-Buffer virtual resource IDs are set to 0 (invalid) — they are
     * only meaningful inside _buildGraph() where addPasses() is called
     * with the per-frame context.
     */
    private _buildPostProcessContext(): PostProcessContext {
        return this._makePostProcessContext(0, 0, 0);
    }

    /**
     * Build a per-frame PostProcessContext with live G-Buffer virtual resource IDs.
     */
    private _makePostProcessContext(
        gbAlbedoAO:        number,
        gbNormalRoughness: number,
        gbMetallicEmissive: number,
    ): PostProcessContext {
        const slot   = this._frameIndex % this._backend.maxFramesInFlight;
        const canvas = this._backend.context.canvas as HTMLCanvasElement;
        const self   = this;
        return {
            backend:            this._backend,
            resources:          this._resources,
            shaderSystem:       this._shaderSystem,
            pipelineManager:    this._pipelineManager,
            gbAlbedoAO,
            gbNormalRoughness,
            gbMetallicEmissive,
            perFrameBuffer:     this._perFrameBuffers[slot]!,
            clampSampler:       this._clampSampler!,
            frameIndex:         this._frameIndex,
            resolution:         [canvas.width, canvas.height],
            resolveVirtualTexture(id, viewDesc) {
                return self._resolveVirtualTexture(id, viewDesc);
            },
            envCubemapTexture: this._envCubemapTexture,
            envCubemapEnabled: this._envCubemapEnabled,
            envCubemapSampler: this._envCubeSampler,
        };
    }

    // -------------------------------------------------------------------------
    // Stats
    // -------------------------------------------------------------------------

    getStats(): Readonly<FrameStats> {
        return this._stats;
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    destroy(): void {
        this.stop();
        for (const buf of this._perFrameBuffers) buf.destroy();
        this._perFrameBuffers.length             = 0;
        this._perFrameBindGroups.length          = 0;
        this._lightingPerFrameBindGroups.length  = 0;
        this._modelMatBuffer?.destroy();
        this._modelMatBuffer            = null;
        this._staticInstanceBuffer?.destroy();
        this._staticInstanceBuffer      = null;
        this._instancedG2BGCache.clear();
        this._gbufferPipelineCache.clear();
        this._gbufferPerFrameBGCache.clear();
        this._gbufferModelMatBGCache.clear();
        this._lightingPipeline          = null;
        this._lightingGBufferBGL        = null;
        this._lightingLightsBGL         = null;
        this._lightingLightsBG          = null;
        this._clampSampler              = null;
        this._blitPipeline              = null;
        this._blitBGL                   = null;
        this._blitExposureBuffer?.destroy();
        this._blitExposureBuffer        = null;
        this._depthDownsamplePipeline                = null;
        this._depthPrepassPipeline                  = null;
        this._depthPrepassInstancedPipeline         = null;
        this._depthPrepassPerFrameBindGroups.length = 0;
        this._depthPrepassModelMatBindGroups.length = 0;
        this._depthPrepassInstancedPerFrameBGs.length = 0;
        this._depthPrepassInstancedG2BG             = null;
        // Shadow resources
        for (const buf of this._shadowVPBuffers) buf.destroy();
        this._shadowVPBuffers.length    = 0;
        this._shadowVPBGs.length        = 0;
        this._shadowVPInstBGs.length    = 0;
        this._shadowModelMatBGs.length  = 0;
        this._shadowPipeline            = null;
        this._shadowInstancedPipeline   = null;
        this._shadowSkinnedPipeline     = null;
        this._shadowVPSkinnedBGs.length = 0;
        this._shadowInstancedG2BG       = null;
        this._shadowCompSampler         = null;
        // Env cubemap resources
        this._envParamsBuffer?.destroy();
        this._envParamsBuffer           = null;
        this._placeholderCubeTex?.destroy();
        this._placeholderCubeTex        = null;
        this._envCubeSampler            = null; // GPUSampler has no destroy()
        this._lightingEnvBGL            = null;
        this._lightingEnvBG             = null;
        this._instanceBatches  = [];
        this._dynamicDrawables = [];
        this._inputSystem               = null;
        this._cameraControllers         = null;
        this._animationSystem           = null;
        // Bind group caches
        this._ddBGCache.clear();
        this._lightingGBufBGCache.clear();
        this._fwdSceneBlitBGCache.clear();
        this._refrDsBGCache.clear();
        this._blitPresentBGCache.clear();
        for (const c of this._fwdPerFrameBGCache.values()) c.clear();
        this._fwdPerFrameBGCache.clear();
        this._skinnedShadowBGPool.clear();
        this._skinnedGBufBGPool.clear();
        this._forwardLinearSampler      = null;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Produce a stable cache key for a set of GBuffer shader defines.
 * Keys are sorted so define order doesn't affect the result.
 * Example: { HAS_TEXTURES: '1', HAS_BASE_COLOR_MAP: '1' } → "HAS_BASE_COLOR_MAP=1|HAS_TEXTURES=1"
 */
function _gbufferVariantKey(defines: ShaderDefines): string {
    return Object.keys(defines).sort().map(k => `${k}=${defines[k]}`).join('|');
}

const _IDENTITY_MAT4 = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
function _identityMat4(): Float32Array {
    return _IDENTITY_MAT4;
}
