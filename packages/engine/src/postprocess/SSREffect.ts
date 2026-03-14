// /src/engine/postprocess/SSREffect.ts
//
// Screen-Space Reflections effect (DDA + cone-traced glossy + cubemap fallback).
//
// Three-pass pipeline:
//   1. Compute pass (ssr_trace.wgsl)       — DDA ray march, output hitUV + confidence + hitDist
//   2. Compute pass (ssr_downsample.wgsl)  — Gaussian downsample HDR into a blurred mip chain
//   3. Render pass  (ssr_composite.wgsl)   — cone-traced glossy sample from mip chain + cubemap fallback

import ssrTraceWGSL      from '../shaders/ssr_trace.wgsl?raw';
import ssrDownsampleWGSL from '../shaders/ssr_downsample.wgsl?raw';
import ssrCompositeWGSL  from '../shaders/ssr_composite.wgsl?raw';
import { PassType, type RenderGraph, type VirtualResourceId } from '../rendergraph/RenderGraph';
import type { PostProcessEffect, PostProcessContext } from './PostProcessStack';
import { HDR_COLOR_FORMAT } from '../pipelines/PipelineManager';

// Trace params: 12 × f32 = 48 bytes (must match SSRParams in ssr_trace.wgsl)
const TRACE_PARAMS_SIZE = 48;
// Composite params: 4 × f32 = 16 bytes (must match SSRCompositeParams)
const COMPOSITE_PARAMS_SIZE = 16;
// Number of mip levels for the blurred colour chain
const MIP_LEVELS = 5;

export class SSREffect implements PostProcessEffect {
    readonly name = 'SSR';
    enabled = true;

    // ── Trace parameters (exposed for editor / scene JSON) ──────────────
    maxRaySteps     = 128;
    thickness       = 0.3;    // view-space thickness (metres)
    stride          = 2.0;    // base pixel stride
    fadeEnd         = 0.85;   // screen-edge fade threshold
    roughnessCutoff = 0.7;    // max roughness (raised: cone tracing handles glossy)
    jitterScale     = 1.0;    // temporal jitter strength
    maxDistance      = 50.0;  // world-space max ray distance
    strideZCutoff    = 0.01;  // adaptive stride depth factor

    // ── Composite parameters ────────────────────────────────────────────
    envFallbackStr  = 0.5;    // cubemap fallback strength (0..1)

    // ── Pooled scratch arrays (avoid per-frame allocations) ───────────
    private _traceParamsScratch = new Float32Array(new ArrayBuffer(TRACE_PARAMS_SIZE));
    private _compParamsScratch  = new Float32Array(new ArrayBuffer(COMPOSITE_PARAMS_SIZE));

    // ── GPU resources ───────────────────────────────────────────────────
    private _tracePipeline:       GPUComputePipeline | null = null;
    private _downsamplePipeline:  GPUComputePipeline | null = null;
    private _compositePipeline:   GPURenderPipeline  | null = null;
    private _traceParamsBuffer:   GPUBuffer | null = null;
    private _compParamsBuffer:    GPUBuffer | null = null;
    private _linearSampler:       GPUSampler | null = null;
    private _mipSampler:          GPUSampler | null = null;
    private _placeholderCubeTex:  GPUTexture | null = null;

    // Cached refs
    private _backend:         PostProcessContext['backend'] | null = null;
    private _shaderSystem:    PostProcessContext['shaderSystem'] | null = null;
    private _pipelineManager: PostProcessContext['pipelineManager'] | null = null;

    // ────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ────────────────────────────────────────────────────────────────────

    init(ctx: PostProcessContext): void {
        this._backend         = ctx.backend;
        this._shaderSystem    = ctx.shaderSystem;
        this._pipelineManager = ctx.pipelineManager;

        // Register shaders
        ctx.shaderSystem.registerSource({ label: 'ssr_trace',      source: ssrTraceWGSL });
        ctx.shaderSystem.registerSource({ label: 'ssr_downsample', source: ssrDownsampleWGSL });
        ctx.shaderSystem.registerSource({ label: 'ssr_composite',  source: ssrCompositeWGSL });

        // Compute pipelines
        const traceVariant = ctx.shaderSystem.getVariant('ssr_trace', {});
        this._tracePipeline = ctx.pipelineManager.getOrCreateComputePipeline({
            computeShader: traceVariant.handle,
            entryPoint:    'cs_main',
        });

        const dsVariant = ctx.shaderSystem.getVariant('ssr_downsample', {});
        this._downsamplePipeline = ctx.pipelineManager.getOrCreateComputePipeline({
            computeShader: dsVariant.handle,
            entryPoint:    'cs_main',
        });

        // Fullscreen render pipeline for compositing
        const compVariant = ctx.shaderSystem.getVariant('ssr_composite', {});
        this._compositePipeline = ctx.pipelineManager.getOrCreateFullscreenPipeline(
            compVariant.handle,
            compVariant.handle,
            HDR_COLOR_FORMAT,
        );

        // Uniform buffers
        this._traceParamsBuffer = ctx.backend.device.createBuffer({
            label: 'SSR/TraceParams',
            size:  TRACE_PARAMS_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._compParamsBuffer = ctx.backend.device.createBuffer({
            label: 'SSR/CompositeParams',
            size:  COMPOSITE_PARAMS_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Linear-clamp sampler (no mip filtering — for trace result sampling)
        this._linearSampler = ctx.backend.device.createSampler({
            label:        'SSR/LinearSampler',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            magFilter:    'linear',
            minFilter:    'linear',
        });

        // Linear-clamp sampler WITH mip filtering (for colour mip chain + cubemap)
        this._mipSampler = ctx.backend.device.createSampler({
            label:        'SSR/MipSampler',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            addressModeW: 'clamp-to-edge',
            magFilter:    'linear',
            minFilter:    'linear',
            mipmapFilter: 'linear',
        });

        // 1×1 placeholder cubemap (black, 6 faces) for when no env cubemap is loaded
        this._placeholderCubeTex = ctx.backend.device.createTexture({
            label:  'SSR/PlaceholderCube',
            size:   { width: 1, height: 1, depthOrArrayLayers: 6 },
            format: 'rgba8unorm',
            usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
    }

    // ────────────────────────────────────────────────────────────────────
    // Per-frame pass declaration
    // ────────────────────────────────────────────────────────────────────

    addPasses(
        graph:      RenderGraph,
        ctx:        PostProcessContext,
        inputColor: VirtualResourceId,
        inputDepth: VirtualResourceId,
        _halfDepth?: VirtualResourceId,
    ): VirtualResourceId {
        if (!this._tracePipeline || !this._downsamplePipeline || !this._compositePipeline ||
            !this._traceParamsBuffer || !this._compParamsBuffer ||
            !this._linearSampler || !this._mipSampler) {
            return inputColor;
        }

        const [w, h] = ctx.resolution;

        // ── Upload trace params (reuse pooled scratch) ─────────────────
        const tp = this._traceParamsScratch;
        tp[0] = this.maxRaySteps; tp[1] = this.thickness;
        tp[2] = this.stride;      tp[3] = this.fadeEnd;
        tp[4] = this.roughnessCutoff; tp[5] = this.jitterScale;
        tp[6] = this.maxDistance;  tp[7] = this.strideZCutoff;
        tp[8] = 0; tp[9] = 0; tp[10] = 0; tp[11] = 0;
        ctx.backend.queue.writeBuffer(this._traceParamsBuffer, 0, tp);

        // ── Upload composite params (reuse pooled scratch) ───────────
        const cp = this._compParamsScratch;
        cp[0] = MIP_LEVELS - 1;
        cp[1] = this.envFallbackStr;
        cp[2] = ctx.envCubemapEnabled ? 1.0 : 0.0;
        cp[3] = 0;
        ctx.backend.queue.writeBuffer(this._compParamsBuffer, 0, cp);

        // ── Declare textures ────────────────────────────────────────────
        const ssrTrace = graph.declareTexture('SSR/Trace', {
            size:   [w, h],
            format: 'rgba16float',
            usage:  GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });

        // Color mip chain starts at half-res (mip 0 = w/2 × h/2).
        // The composite reads hdrColor for mirror-perfect reflections,
        // and this chain for glossy lookups at progressively lower res.
        const mipW0 = Math.max(1, w >> 1);
        const mipH0 = Math.max(1, h >> 1);
        const colorMipChain = graph.declareTexture('SSR/ColorMips', {
            size:       [mipW0, mipH0],
            format:     HDR_COLOR_FORMAT,
            usage:      GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
            mipLevelCount: MIP_LEVELS,
        });

        const outputColor = graph.declareTexture('SSR/Output', {
            size:   [w, h],
            format: HDR_COLOR_FORMAT,
            usage:  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        // Capture references for closures
        const tracePipeline      = this._tracePipeline;
        const downsamplePipeline = this._downsamplePipeline;
        const compositePipeline  = this._compositePipeline;
        const traceParamsBuffer  = this._traceParamsBuffer;
        const compParamsBuffer   = this._compParamsBuffer;
        const linearSampler      = this._linearSampler;
        const mipSampler         = this._mipSampler;
        const perFrameBuffer     = ctx.perFrameBuffer;
        const gbNormalRoughness  = ctx.gbNormalRoughness;
        const gbMetallicEmissive = ctx.gbMetallicEmissive;
        // Resolve env cubemap texture
        const cubeTex = (ctx.envCubemapEnabled && ctx.envCubemapTexture)
            ? ctx.envCubemapTexture
            : this._placeholderCubeTex!;

        // ── Pass 1: SSR Trace (compute) ─────────────────────────────────

        graph.addPass({
            name:           'SSR/Trace',
            type:           PassType.Compute,
            reads:          [inputDepth, gbNormalRoughness],
            writes:         [ssrTrace],
            hasSideEffects: false,
            execute: (_encoder, passEncoder) => {
                if (!passEncoder) return;

                const normalView   = ctx.resolveVirtualTexture(gbNormalRoughness);
                const depthView    = ctx.resolveVirtualTexture(inputDepth, { aspect: 'depth-only' });
                const ssrView      = ctx.resolveVirtualTexture(ssrTrace);
                if (!normalView || !depthView || !ssrView) return;

                const bg = ctx.backend.device.createBindGroup({
                    label:  'SSR/Trace/BG',
                    layout: tracePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: perFrameBuffer } },
                        { binding: 1, resource: normalView },
                        { binding: 2, resource: depthView },
                        { binding: 3, resource: { buffer: traceParamsBuffer } },
                        { binding: 4, resource: ssrView },
                    ],
                });

                const cp = passEncoder as GPUComputePassEncoder;
                cp.setPipeline(tracePipeline);
                cp.setBindGroup(0, bg);
                cp.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
            },
        });

        // ── Pass 2: Downsample mip chain (single pass, multiple dispatches) ──
        // All mip levels are generated in one compute pass to avoid render
        // graph cycles (same resource in both reads and writes).
        // Dispatch 0: inputColor (full-res) → colorMipChain mip 0 (half-res)
        // Dispatch 1: colorMipChain mip 0   → colorMipChain mip 1 (quarter-res)
        // ...

        graph.addPass({
            name:           'SSR/Downsample',
            type:           PassType.Compute,
            reads:          [inputColor],
            writes:         [colorMipChain],
            hasSideEffects: true,
            execute: (_encoder, passEncoder) => {
                if (!passEncoder) return;
                const cp = passEncoder as GPUComputePassEncoder;
                cp.setPipeline(downsamplePipeline);

                for (let dstMip = 0; dstMip < MIP_LEVELS; dstMip++) {
                    const mipW = Math.max(1, w >> (dstMip + 1));
                    const mipH = Math.max(1, h >> (dstMip + 1));

                    // Source: inputColor for first dispatch, previous mip for subsequent
                    let srcView: GPUTextureView | undefined;
                    if (dstMip === 0) {
                        srcView = ctx.resolveVirtualTexture(inputColor);
                    } else {
                        srcView = ctx.resolveVirtualTexture(colorMipChain, {
                            baseMipLevel:  dstMip - 1,
                            mipLevelCount: 1,
                        });
                    }

                    const dstView = ctx.resolveVirtualTexture(colorMipChain, {
                        baseMipLevel:  dstMip,
                        mipLevelCount: 1,
                    });

                    if (!srcView || !dstView) continue;

                    const bg = ctx.backend.device.createBindGroup({
                        label:  `SSR/Downsample/BG/Mip${dstMip}`,
                        layout: downsamplePipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: srcView },
                            { binding: 1, resource: dstView },
                        ],
                    });

                    cp.setBindGroup(0, bg);
                    cp.dispatchWorkgroups(Math.ceil(mipW / 8), Math.ceil(mipH / 8));
                }
            },
        });

        // ── Pass N+1: SSR Composite (render) ────────────────────────────

        graph.addPass({
            name:  'SSR/Composite',
            type:  PassType.Render,
            reads: [inputColor, ssrTrace, colorMipChain, gbNormalRoughness, gbMetallicEmissive, inputDepth],
            colorAttachments: [{
                resourceId: outputColor,
                loadOp:     'clear',
                storeOp:    'store',
                clearColor: { r: 0, g: 0, b: 0, a: 1 },
            }],
            hasSideEffects: false,
            execute: (_encoder, passEncoder) => {
                if (!passEncoder) return;

                const hdrView      = ctx.resolveVirtualTexture(inputColor);
                const traceView    = ctx.resolveVirtualTexture(ssrTrace);
                const mipView      = ctx.resolveVirtualTexture(colorMipChain);
                const normalView   = ctx.resolveVirtualTexture(gbNormalRoughness);
                const metallicView = ctx.resolveVirtualTexture(gbMetallicEmissive);
                const depthView    = ctx.resolveVirtualTexture(inputDepth, { aspect: 'depth-only' });
                if (!hdrView || !traceView || !mipView || !normalView || !metallicView || !depthView) return;

                const cubeView = cubeTex.createView({ dimension: 'cube', arrayLayerCount: 6 });

                const bg = ctx.backend.device.createBindGroup({
                    label:  'SSR/Composite/BG',
                    layout: compositePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: hdrView },
                        { binding: 1, resource: traceView },
                        { binding: 2, resource: mipView },
                        { binding: 3, resource: normalView },
                        { binding: 4, resource: metallicView },
                        { binding: 5, resource: depthView },
                        { binding: 6, resource: cubeView },
                        { binding: 7, resource: mipSampler },
                        { binding: 8, resource: { buffer: perFrameBuffer } },
                        { binding: 9, resource: { buffer: compParamsBuffer } },
                    ],
                });

                const rp = passEncoder as GPURenderPassEncoder;
                rp.setPipeline(compositePipeline);
                rp.setBindGroup(0, bg);
                rp.draw(3);
            },
        });

        return outputColor;
    }

    // ────────────────────────────────────────────────────────────────────
    // Cleanup
    // ────────────────────────────────────────────────────────────────────

    destroy(): void {
        this._traceParamsBuffer?.destroy();
        this._compParamsBuffer?.destroy();
        this._placeholderCubeTex?.destroy();
        this._traceParamsBuffer    = null;
        this._compParamsBuffer     = null;
        this._tracePipeline        = null;
        this._downsamplePipeline   = null;
        this._compositePipeline    = null;
        this._linearSampler        = null;
        this._mipSampler           = null;
        this._placeholderCubeTex   = null;
    }
}
