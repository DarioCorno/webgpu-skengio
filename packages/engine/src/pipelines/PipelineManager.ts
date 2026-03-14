// /src/engine/pipelines/PipelineManager.ts

import { Logger } from '../core/Logger';
import { VertexSemantic, type VertexLayoutDesc } from '../geometry/MeshSystem';
import type { GPUBackend } from '../core/GPUBackend';
import type { ShaderSystem, ShaderHandle } from '../shaders/ShaderSystem';

// -------------------------------------------------------------------------
// Standard semantic → shader location table
// -------------------------------------------------------------------------

/**
 * Canonical @location binding for every VertexSemantic.
 * All engine shaders must use these locations for vertex inputs.
 *
 *   @location(0) position : vec3<f32>
 *   @location(1) normal   : vec3<f32>
 *   @location(2) tangent  : vec4<f32>
 *   @location(3) uv0      : vec2<f32>
 *   @location(4) uv1      : vec2<f32>
 *   @location(5) color0   : vec4<f32>
 *   @location(6) joints0  : vec4<u32>
 *   @location(7) weights0 : vec4<f32>
 */
export const SEMANTIC_LOCATION: Record<VertexSemantic, number> = {
    [VertexSemantic.Position]: 0,
    [VertexSemantic.Normal]:   1,
    [VertexSemantic.Tangent]:  2,
    [VertexSemantic.UV0]:      3,
    [VertexSemantic.UV1]:      4,
    [VertexSemantic.Color0]:   5,
    [VertexSemantic.Joints0]:  6,
    [VertexSemantic.Weights0]: 7,
};

/**
 * Convert engine VertexLayoutDesc[] (semantic-based) to the
 * GPUVertexBufferLayout[] that a pipeline descriptor expects.
 *
 * The mapping VertexSemantic → @location is taken from SEMANTIC_LOCATION.
 */
export function buildVertexBufferLayouts(layouts: VertexLayoutDesc[]): GPUVertexBufferLayout[] {
    return layouts.map(layout => ({
        arrayStride: layout.arrayStride,
        stepMode:    layout.stepMode ?? 'vertex',
        attributes:  layout.attributes.map(attr => ({
            format:         attr.format,
            offset:         attr.offset,
            shaderLocation: SEMANTIC_LOCATION[attr.semantic],
        })),
    }));
}

// -------------------------------------------------------------------------
// G-Buffer formats (shared constant used by passes and pipeline factories)
// -------------------------------------------------------------------------

/**
 * The three MRT color-attachment formats for the G-buffer geometry pass.
 *
 *   RT0  rgba8unorm      — albedo RGB + occlusion A
 *   RT1  rgba16float     — world-space normal XYZ (packed) + roughness A
 *   RT2  rgba8unorm      — metallic R + emissive GBA
 */
export const GBUFFER_COLOR_FORMATS: GPUTextureFormat[] = [
    'rgba8unorm',    // albedo + occlusion
    'rgba16float',   // normal + roughness
    'rgba8unorm',    // metallic + emissive
] as const;

export const GBUFFER_DEPTH_FORMAT: GPUTextureFormat = 'depth32float';
export const SHADOW_DEPTH_FORMAT:  GPUTextureFormat = 'depth32float';
export const HDR_COLOR_FORMAT:     GPUTextureFormat = 'rgba16float';

// -------------------------------------------------------------------------
// Pipeline key types
// -------------------------------------------------------------------------

/**
 * Everything needed to uniquely identify and create a render pipeline.
 */
export interface RenderPipelineKey {
    vertexShader:    ShaderHandle;
    fragmentShader?: ShaderHandle;            // optional — omit for depth-only passes
    vertexEntryPoint?:   string;              // default: 'vs_main'
    fragmentEntryPoint?: string;              // default: 'fs_main'
    vertexBufferLayouts: GPUVertexBufferLayout[];
    colorTargetFormats:  GPUTextureFormat[];   // empty array = depth-only pass
    depthStencilFormat?: GPUTextureFormat;
    blendStates?:        (GPUBlendState | undefined)[];
    cullMode?:           GPUCullMode;          // default: 'back'
    topology?:           GPUPrimitiveTopology; // default: 'triangle-list'
    sampleCount?:        number;               // default: 1
    depthWriteEnabled?:  boolean;              // default: true
    depthCompare?:       GPUCompareFunction;   // default: 'less'
    /**
     * Integer depth bias added to each fragment's depth value before the depth
     * comparison.  Negative values push the fragment closer to the camera.
     * Use a small negative value (e.g. -2) on the G-Buffer pass when a depth
     * prepass is active, to ensure the G-Buffer depth always passes 'less-equal'
     * against the prepass value despite 1–2 ULP FP differences between the two
     * separately-compiled pipeline variants.
     */
    depthBias?:          number;               // default: 0
    /** Strip index format — only needed for strip topologies. */
    stripIndexFormat?:   GPUIndexFormat;
}

export interface ComputePipelineKey {
    computeShader: ShaderHandle;
    entryPoint:    string;
}

/** Future: ray-tracing pipeline (Tier2_RT). */
export interface RayTracingPipelineKey {
    rayGenShader:      ShaderHandle;
    missShaders:       ShaderHandle[];
    closestHitShaders: ShaderHandle[];
}

export type PipelineHandle = number;

// -------------------------------------------------------------------------
// Standard blend presets
// -------------------------------------------------------------------------

/** Pre-multiplied alpha blending. */
export const BLEND_PREMULTIPLIED_ALPHA: GPUBlendState = {
    color: { srcFactor: 'one',            dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one',            dstFactor: 'one-minus-src-alpha', operation: 'add' },
};

/** Straight (un-premultiplied) alpha blending. */
export const BLEND_STRAIGHT_ALPHA: GPUBlendState = {
    color: { srcFactor: 'src-alpha',      dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one',            dstFactor: 'one-minus-src-alpha', operation: 'add' },
};

/** Additive blending (particles, emissive effects). */
export const BLEND_ADDITIVE: GPUBlendState = {
    color: { srcFactor: 'src-alpha',      dstFactor: 'one',                 operation: 'add' },
    alpha: { srcFactor: 'zero',           dstFactor: 'one',                 operation: 'add' },
};

// -------------------------------------------------------------------------
// PipelineManager
// -------------------------------------------------------------------------

/**
 * Creates and caches render / compute pipelines.
 *
 * Every pipeline is uniquely identified by a JSON-hashed key.  The manager
 * returns a cached GPU pipeline when the same key is requested again.
 *
 * Convenience factory methods encode the correct key for each deferred-
 * renderer pass so callers don't need to assemble keys by hand:
 *
 *   getOrCreateGBufferPipeline(vs, fs, layouts)
 *   getOrCreateShadowPipeline(vs, layouts)
 *   getOrCreateFullscreenPipeline(vs, fs, colorFormat)
 *   getOrCreateForwardPipeline(vs, fs, layouts, blendState)
 *   getOrCreateComputePipeline(key)
 */
export class PipelineManager {

    private _backend!: GPUBackend;
    private _shaderSystem!: ShaderSystem;
    private _nextHandle: PipelineHandle = 1;
    private readonly _log = new Logger('PipelineManager');

    private _renderPipelineCache:  Map<string, { handle: PipelineHandle; pipeline: GPURenderPipeline }>  = new Map();
    private _computePipelineCache: Map<string, { handle: PipelineHandle; pipeline: GPUComputePipeline }> = new Map();

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    init(backend: GPUBackend, shaderSystem: ShaderSystem): void {
        this._backend = backend;
        this._shaderSystem = shaderSystem;
    }

    // -------------------------------------------------------------------------
    // Core — render pipelines
    // -------------------------------------------------------------------------

    /**
     * Return a cached render pipeline, or create + cache a new one synchronously.
     * Prefer the async variant for non-blocking pipeline compilation.
     */
    getOrCreateRenderPipeline(key: RenderPipelineKey): GPURenderPipeline {
        const hash   = this._hashRenderKey(key);
        const cached = this._renderPipelineCache.get(hash);
        if (cached) return cached.pipeline;

        const desc     = this._buildRenderPipelineDescriptor(key);
        const pipeline = this._backend.device.createRenderPipeline(desc);
        const handle   = this._nextHandle++;

        this._renderPipelineCache.set(hash, { handle, pipeline });
        this._log.info(`Created render pipeline handle=${handle} topology=${key.topology ?? 'triangle-list'} ` +
                       `colorTargets=${key.colorTargetFormats.length} depth=${key.depthStencilFormat ?? 'none'}`);
        return pipeline;
    }

    /**
     * Async variant — uses createRenderPipelineAsync so pipeline compilation
     * does not block the main thread. Preferred for load-time compilation.
     */
    async getOrCreateRenderPipelineAsync(key: RenderPipelineKey): Promise<GPURenderPipeline> {
        const hash   = this._hashRenderKey(key);
        const cached = this._renderPipelineCache.get(hash);
        if (cached) return cached.pipeline;

        const desc     = this._buildRenderPipelineDescriptor(key);
        const pipeline = await this._backend.device.createRenderPipelineAsync(desc);
        const handle   = this._nextHandle++;

        this._renderPipelineCache.set(hash, { handle, pipeline });
        this._log.info(`Created render pipeline (async) handle=${handle}`);
        return pipeline;
    }

    // -------------------------------------------------------------------------
    // Core — compute pipelines
    // -------------------------------------------------------------------------

    getOrCreateComputePipeline(key: ComputePipelineKey): GPUComputePipeline {
        const hash   = this._hashComputeKey(key);
        const cached = this._computePipelineCache.get(hash);
        if (cached) return cached.pipeline;

        const variant = this._shaderSystem.getVariantByHandle(key.computeShader);
        if (!variant) throw new Error(`[PipelineManager] Compute shader handle ${key.computeShader} not found`);

        const pipeline = this._backend.device.createComputePipeline({
            label:   `cp_${this._nextHandle}`,
            layout:  'auto',
            compute: { module: variant.module, entryPoint: key.entryPoint },
        });
        const handle = this._nextHandle++;

        this._computePipelineCache.set(hash, { handle, pipeline });
        this._log.info(`Created compute pipeline handle=${handle} entry=${key.entryPoint}`);
        return pipeline;
    }

    async getOrCreateComputePipelineAsync(key: ComputePipelineKey): Promise<GPUComputePipeline> {
        const hash   = this._hashComputeKey(key);
        const cached = this._computePipelineCache.get(hash);
        if (cached) return cached.pipeline;

        const variant = this._shaderSystem.getVariantByHandle(key.computeShader);
        if (!variant) throw new Error(`[PipelineManager] Compute shader handle ${key.computeShader} not found`);

        const pipeline = await this._backend.device.createComputePipelineAsync({
            label:   `cp_${this._nextHandle}`,
            layout:  'auto',
            compute: { module: variant.module, entryPoint: key.entryPoint },
        });
        const handle = this._nextHandle++;

        this._computePipelineCache.set(hash, { handle, pipeline });
        this._log.info(`Created compute pipeline (async) handle=${handle}`);
        return pipeline;
    }

    // -------------------------------------------------------------------------
    // Deferred-renderer pass factories
    // -------------------------------------------------------------------------

    /**
     * G-buffer geometry fill pass.
     *
     * Outputs:   GBUFFER_COLOR_FORMATS (3 MRTs) + GBUFFER_DEPTH_FORMAT
     * Depth:     write-enabled, compare=less  (or equal/no-write when depthPrepass=true)
     * Cull:      back-face (override with doubleSided=true to disable)
     *
     * When `depthPrepass` is true the pass assumes the depth buffer was already
     * filled by a prior depth-only prepass.  The compare function switches to
     * 'less-equal' and depth writes are disabled so the GPU early-Z unit can
     * discard fragments that are behind the prepass depth without invoking the
     * expensive fragment shader.
     *
     * NOTE: 'less-equal' rather than 'equal' is intentional.  The depth-only
     * prepass and the full G-Buffer pipeline are compiled separately; even with
     * identical WGSL vertex code the driver may produce 1–2 ULP differences in
     * the computed clip-space Z.  'equal' would reject those fragments at
     * silhouette edges (steep depth gradient, nearly-edge-on triangles),
     * leaving holes in the G-Buffer that render as white borders in the
     * lighting pass.  'less-equal' is equally effective for early-Z rejection
     * of fully occluded geometry and is immune to the ULP rounding difference.
     */
    getOrCreateGBufferPipeline(
        vsHandle: ShaderHandle,
        fsHandle: ShaderHandle,
        vertexLayouts: VertexLayoutDesc[],
        options: { doubleSided?: boolean; alphaMask?: boolean; depthPrepass?: boolean } = {},
    ): GPURenderPipeline {
        return this.getOrCreateRenderPipeline({
            vertexShader:        vsHandle,
            fragmentShader:      fsHandle,
            vertexBufferLayouts: buildVertexBufferLayouts(vertexLayouts),
            colorTargetFormats:  GBUFFER_COLOR_FORMATS as GPUTextureFormat[],
            depthStencilFormat:  GBUFFER_DEPTH_FORMAT,
            cullMode:            options.doubleSided ? 'none' : 'back',
            depthWriteEnabled:   !options.depthPrepass,
            depthCompare:        options.depthPrepass  ? 'less-equal'  :
                                 options.alphaMask     ? 'less-equal'  : 'less',
            // When the depth prepass is active, push the G-Buffer depth 2 units
            // closer so that driver-level FP differences between the depth-only
            // and full-MRT pipeline compilations never cause a false 'less-equal'
            // failure at silhouette edges.
            depthBias:           options.depthPrepass  ? -2            : 0,
        });
    }

    /**
     * Depth-only prepass — no colour attachments, vertex stage only.
     *
     * Writes the full scene depth in a single inexpensive pass so the
     * subsequent G-Buffer fill can use depthCompare='equal' to benefit from
     * hardware early-Z rejection of occluded fragments.
     *
     * Bind groups expected by the shader:
     *   @group(0)  PerFrameUniforms  (view/proj matrices)
     *   @group(2)  ModelUniforms     (per-draw model matrix)
     */
    getOrCreateDepthPrepassPipeline(
        vsHandle: ShaderHandle,
        vertexLayouts: VertexLayoutDesc[],
        options: { doubleSided?: boolean } = {},
    ): GPURenderPipeline {
        return this.getOrCreateRenderPipeline({
            vertexShader:        vsHandle,
            // No fragment shader → depth-only pass.
            vertexBufferLayouts: buildVertexBufferLayouts(vertexLayouts),
            colorTargetFormats:  [],               // no colour output
            depthStencilFormat:  GBUFFER_DEPTH_FORMAT,
            cullMode:            options.doubleSided ? 'none' : 'back',
            depthWriteEnabled:   true,
            depthCompare:        'less',
        });
    }

    /**
     * Shadow depth-only pass.
     *
     * No color attachments — writes only to depth.
     * Uses a slight positive depth bias to reduce shadow acne.
     */
    getOrCreateShadowPipeline(
        vsHandle: ShaderHandle,
        vertexLayouts: VertexLayoutDesc[],
    ): GPURenderPipeline {
        return this.getOrCreateRenderPipeline({
            vertexShader:        vsHandle,
            vertexBufferLayouts: buildVertexBufferLayouts(vertexLayouts),
            colorTargetFormats:  [],
            depthStencilFormat:  SHADOW_DEPTH_FORMAT,
            cullMode:            'back',  // render faces toward the light; bias prevents acne
            depthWriteEnabled:   true,
            depthCompare:        'less',
        });
    }

    /**
     * Deferred lighting pass — fullscreen triangle, reads G-buffer, outputs HDR.
     *
     * No vertex buffers. The vertex shader generates a clip-space triangle
     * from gl_VertexIndex without any bound geometry.
     */
    getOrCreateDeferredLightingPipeline(
        vsHandle: ShaderHandle,
        fsHandle: ShaderHandle,
        hdrFormat: GPUTextureFormat = HDR_COLOR_FORMAT,
    ): GPURenderPipeline {
        return this.getOrCreateRenderPipeline({
            vertexShader:        vsHandle,
            fragmentShader:      fsHandle,
            vertexBufferLayouts: [],        // fullscreen triangle: no vertex buffer
            colorTargetFormats:  [hdrFormat],
            cullMode:            'none',
            depthWriteEnabled:   false,     // no depth involvement in lighting pass
        });
    }

    /**
     * Forward transparency pass.
     *
     * Reads depth from G-buffer (no write), alpha-blends onto HDR.
     * Sort objects back-to-front before drawing.
     */
    getOrCreateForwardPipeline(
        vsHandle: ShaderHandle,
        fsHandle: ShaderHandle,
        vertexLayouts: VertexLayoutDesc[],
        options: {
            blend?:      GPUBlendState;
            hdrFormat?:  GPUTextureFormat;
            doubleSided?: boolean;
        } = {},
    ): GPURenderPipeline {
        const blend  = options.blend     ?? BLEND_PREMULTIPLIED_ALPHA;
        const format = options.hdrFormat ?? HDR_COLOR_FORMAT;

        return this.getOrCreateRenderPipeline({
            vertexShader:        vsHandle,
            fragmentShader:      fsHandle,
            vertexBufferLayouts: buildVertexBufferLayouts(vertexLayouts),
            colorTargetFormats:  [format],
            blendStates:         [blend],
            depthStencilFormat:  GBUFFER_DEPTH_FORMAT,
            cullMode:            options.doubleSided ? 'none' : 'back',
            depthWriteEnabled:   false,    // transparent objects don't write depth
            depthCompare:        'less',
        });
    }

    /**
     * Generic fullscreen post-process pass.
     *
     * No vertex buffer — vertex shader generates the triangle.
     * Suitable for tone-mapping, FXAA, bloom, TAA resolve, etc.
     */
    getOrCreateFullscreenPipeline(
        vsHandle: ShaderHandle,
        fsHandle: ShaderHandle,
        colorFormat: GPUTextureFormat,
        blend?: GPUBlendState,
    ): GPURenderPipeline {
        return this.getOrCreateRenderPipeline({
            vertexShader:        vsHandle,
            fragmentShader:      fsHandle,
            vertexBufferLayouts: [],
            colorTargetFormats:  [colorFormat],
            blendStates:         blend ? [blend] : undefined,
            cullMode:            'none',
            depthWriteEnabled:   false,
        });
    }

    // -------------------------------------------------------------------------
    // Cache management
    // -------------------------------------------------------------------------

    /** Evict all cached pipelines — call after a shader hot-reload. */
    invalidateAll(): void {
        this._renderPipelineCache.clear();
        this._computePipelineCache.clear();
        this._log.debug('All pipeline caches cleared');
    }

    destroy(): void {
        this.invalidateAll();
    }

    // -------------------------------------------------------------------------
    // Internal — descriptor builder
    // -------------------------------------------------------------------------

    private _buildRenderPipelineDescriptor(key: RenderPipelineKey): GPURenderPipelineDescriptor {
        // --- Resolve shader handles to GPUShaderModule -----------------------
        const vsVariant = this._shaderSystem.getVariantByHandle(key.vertexShader);
        if (!vsVariant) {
            throw new Error(`[PipelineManager] Vertex shader handle ${key.vertexShader} not found`);
        }

        let fsVariant = key.fragmentShader !== undefined
            ? this._shaderSystem.getVariantByHandle(key.fragmentShader)
            : undefined;
        if (key.fragmentShader !== undefined && !fsVariant) {
            throw new Error(`[PipelineManager] Fragment shader handle ${key.fragmentShader} not found`);
        }

        // --- Color targets ---------------------------------------------------
        const colorTargets: GPUColorTargetState[] = key.colorTargetFormats.map((format, i) => ({
            format,
            writeMask: GPUColorWrite.ALL,
            blend:     key.blendStates?.[i],
        }));

        // --- Depth-stencil ---------------------------------------------------
        const depthStencil: GPUDepthStencilState | undefined = key.depthStencilFormat
            ? {
                format:            key.depthStencilFormat,
                depthWriteEnabled: key.depthWriteEnabled ?? true,
                depthCompare:      key.depthCompare      ?? 'less',
                depthBias:         key.depthBias         ?? 0,
            }
            : undefined;

        // --- Primitive state -------------------------------------------------
        const topology = key.topology ?? 'triangle-list';
        const primitive: GPUPrimitiveState = {
            topology,
            frontFace:        'ccw',
            cullMode:          key.cullMode ?? 'back',
            stripIndexFormat:  key.stripIndexFormat,
        };

        // --- Assemble descriptor ---------------------------------------------
        const desc: GPURenderPipelineDescriptor = {
            label:  `rp_${this._nextHandle}`,
            layout: 'auto',
            vertex: {
                module:     vsVariant.module,
                entryPoint: key.vertexEntryPoint ?? 'vs_main',
                buffers:    key.vertexBufferLayouts,
            },
            primitive,
            depthStencil,
            multisample: { count: key.sampleCount ?? 1 },
        };

        // Fragment stage is optional (depth-only passes have no color targets)
        if (fsVariant) {
            desc.fragment = {
                module:     fsVariant.module,
                entryPoint: key.fragmentEntryPoint ?? 'fs_main',
                targets:    colorTargets,
            };
        } else if (colorTargets.length > 0) {
            // Caller asked for color outputs but provided no fragment shader — warn
            this._log.warn('Color targets specified but no fragment shader provided — targets ignored');
        }

        return desc;
    }

    // -------------------------------------------------------------------------
    // Hashing
    // -------------------------------------------------------------------------

    private _hashRenderKey(key: RenderPipelineKey): string {
        return JSON.stringify(key);
    }

    private _hashComputeKey(key: ComputePipelineKey): string {
        return JSON.stringify(key);
    }
}
