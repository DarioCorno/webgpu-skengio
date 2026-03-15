// /src/engine/postprocess/SSAOEffect.ts
//
// Screen-Space Ambient Occlusion effect.
//
// Three passes:
//   1. Compute pass (ssao_compute.wgsl)    — hemisphere sampling at half-res
//   2. Compute pass (ssao_blur.wgsl)       — bilateral depth-aware blur at half-res
//   3. Render pass  (ssao_composite.wgsl)  — bilinear upscale and multiply into scene

import ssaoComputeWGSL   from '../shaders/ssao_compute.wgsl?raw';
import ssaoBlurWGSL      from '../shaders/ssao_blur.wgsl?raw';
import ssaoCompositeWGSL from '../shaders/ssao_composite.wgsl?raw';
import { PassType, type RenderGraph, type VirtualResourceId } from '../rendergraph/RenderGraph';
import type { PostProcessEffect, PostProcessContext } from './PostProcessStack';
import { HDR_COLOR_FORMAT } from '../pipelines/PipelineManager';
import { BindGroupCache } from '../core/BindGroupCache';

// SSAOParams uniform layout (must match SSAOParams in ssao_compute.wgsl / ssao_blur.wgsl)
const SSAO_PARAMS_SIZE = 32; // 8 × f32
const MAX_SAMPLES = 32;
const GOLDEN_ANGLE = 2.399963;

/** Precompute hemisphere sample directions (golden-angle spiral). */
function buildSampleKernel(count: number): Float32Array<ArrayBuffer> {
    // Each sample is vec4f (xyz direction, w unused/padding) = 16 bytes
    const data = new Float32Array(new ArrayBuffer(count * 4 * 4));
    for (let i = 0; i < count; i++) {
        const fi = i;
        const r = Math.sqrt((fi + 0.5) / count);
        const angle = fi * GOLDEN_ANGLE;
        const x = Math.cos(angle) * r;
        const y = Math.sin(angle) * r;
        const h = r;
        // Normalize the hemisphere sample
        const len = Math.sqrt(x * x + y * y + h * h) || 1;
        data[i * 4 + 0] = x / len;
        data[i * 4 + 1] = y / len;
        data[i * 4 + 2] = h / len;
        data[i * 4 + 3] = r; // store radius for weighting
    }
    return data;
}

export class SSAOEffect implements PostProcessEffect {
    readonly name = 'SSAO';
    enabled = true;

    // Tuning parameters (exposed for editor / scene JSON)
    radius        = 0.5;    // world-space hemisphere radius
    bias          = 0.02;   // depth bias to avoid self-occlusion
    intensity     = 1.5;    // AO strength multiplier
    sampleCount   = 16;     // samples per pixel (8, 12, or 16)
    blurSharpness = 10.0;   // depth-edge sensitivity for bilateral blur

    // GPU resources (created once in init, reused each frame)
    private _computePipeline:   GPUComputePipeline | null = null;
    private _blurPipeline:      GPUComputePipeline | null = null;
    private _compositePipeline: GPURenderPipeline  | null = null;
    private _paramsBuffer:      GPUBuffer | null = null;
    private _kernelBuffer:      GPUBuffer | null = null;
    private _linearSampler:     GPUSampler | null = null;
    private _lastKernelCount:   number = 0;
    private _paramsScratch      = new Float32Array(new ArrayBuffer(SSAO_PARAMS_SIZE));

    // Cached refs
    private _backend:         PostProcessContext['backend'] | null = null;

    // Bind group caches (avoid per-frame createBindGroup allocations)
    private _computeBGCache  = new BindGroupCache();
    private _blurBGCache     = new BindGroupCache();
    private _compositeBGCache = new BindGroupCache();

    init(ctx: PostProcessContext): void {
        this._backend = ctx.backend;

        // Register shaders
        ctx.shaderSystem.registerSource({ label: 'ssao_compute',   source: ssaoComputeWGSL });
        ctx.shaderSystem.registerSource({ label: 'ssao_blur',      source: ssaoBlurWGSL });
        ctx.shaderSystem.registerSource({ label: 'ssao_composite', source: ssaoCompositeWGSL });

        // Compute pipeline: AO calculation
        const computeVariant = ctx.shaderSystem.getVariant('ssao_compute', {});
        this._computePipeline = ctx.pipelineManager.getOrCreateComputePipeline({
            computeShader: computeVariant.handle,
            entryPoint:    'cs_main',
        });

        // Compute pipeline: bilateral blur
        const blurVariant = ctx.shaderSystem.getVariant('ssao_blur', {});
        this._blurPipeline = ctx.pipelineManager.getOrCreateComputePipeline({
            computeShader: blurVariant.handle,
            entryPoint:    'cs_main',
        });

        // Fullscreen render pipeline: composite
        const compVariant = ctx.shaderSystem.getVariant('ssao_composite', {});
        this._compositePipeline = ctx.pipelineManager.getOrCreateFullscreenPipeline(
            compVariant.handle,
            compVariant.handle,
            HDR_COLOR_FORMAT,
        );

        // Params uniform buffer
        this._paramsBuffer = ctx.backend.device.createBuffer({
            label: 'SSAO/Params',
            size:  SSAO_PARAMS_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Precomputed sample kernel (storage buffer, uploaded when sample count changes)
        this._kernelBuffer = ctx.backend.device.createBuffer({
            label: 'SSAO/Kernel',
            size:  MAX_SAMPLES * 16, // vec4f per sample
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Linear-clamp sampler for bilinear upscale in composite pass
        this._linearSampler = ctx.backend.device.createSampler({
            label:        'SSAO/LinearSampler',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            magFilter:    'linear',
            minFilter:    'linear',
        });
    }

    addPasses(
        graph:      RenderGraph,
        ctx:        PostProcessContext,
        inputColor: VirtualResourceId,
        inputDepth: VirtualResourceId,
        halfDepth?: VirtualResourceId,
    ): VirtualResourceId {
        if (!this._computePipeline || !this._blurPipeline || !this._compositePipeline ||
            !this._paramsBuffer || !this._kernelBuffer || !this._linearSampler || halfDepth === undefined) {
            return inputColor;
        }

        const [w, h] = ctx.resolution;
        const halfW = Math.max(1, Math.ceil(w / 2));
        const halfH = Math.max(1, Math.ceil(h / 2));

        // Upload SSAO params (reuse pooled scratch)
        const paramsData = this._paramsScratch;
        paramsData[0] = this.radius;     paramsData[1] = this.bias;
        paramsData[2] = this.intensity;  paramsData[3] = this.sampleCount;
        paramsData[4] = this.blurSharpness;
        paramsData[5] = 0; paramsData[6] = 0; paramsData[7] = 0;
        ctx.backend.queue.writeBuffer(this._paramsBuffer, 0, paramsData);

        // Upload precomputed sample kernel when sample count changes
        if (this._kernelBuffer && this.sampleCount !== this._lastKernelCount) {
            const count = Math.min(Math.max(1, this.sampleCount), MAX_SAMPLES);
            const kernel = buildSampleKernel(count);
            ctx.backend.queue.writeBuffer(this._kernelBuffer, 0, kernel);
            this._lastKernelCount = this.sampleCount;
        }

        // Declare transient half-res AO textures
        const HALF_AO_USAGE = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;

        const aoRaw = graph.declareTexture('SSAO/Raw', {
            size:   [halfW, halfH],
            format: 'rgba16float',
            usage:  HALF_AO_USAGE,
        });

        const aoBlurred = graph.declareTexture('SSAO/Blurred', {
            size:   [halfW, halfH],
            format: 'rgba16float',
            usage:  HALF_AO_USAGE,
        });

        // Declare output colour texture
        const outputColor = graph.declareTexture('SSAO/Output', {
            size:   [w, h],
            format: HDR_COLOR_FORMAT,
            usage:  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        // ---- Pass 1: SSAO Compute ------------------------------------------------

        const computePipeline  = this._computePipeline;
        const paramsBuffer     = this._paramsBuffer;
        const kernelBuffer     = this._kernelBuffer!;
        const perFrameBuffer   = ctx.perFrameBuffer;
        const gbNormalRoughness = ctx.gbNormalRoughness;

        graph.addPass({
            name:           'SSAO/Compute',
            type:           PassType.Compute,
            reads:          [halfDepth, gbNormalRoughness],
            writes:         [aoRaw],
            hasSideEffects: false,
            execute: (_encoder, passEncoder) => {
                if (!passEncoder) return;

                const depthView  = ctx.resolveVirtualTexture(halfDepth);
                const normalView = ctx.resolveVirtualTexture(gbNormalRoughness);
                const aoView     = ctx.resolveVirtualTexture(aoRaw);
                if (!depthView || !normalView || !aoView) return;

                const bg = this._computeBGCache.getOrCreate(
                    [depthView, normalView, aoView],
                    () => ctx.backend.device.createBindGroup({
                        label:  'SSAO/Compute/BG',
                        layout: computePipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: perFrameBuffer } },
                            { binding: 1, resource: depthView },
                            { binding: 2, resource: normalView },
                            { binding: 3, resource: { buffer: paramsBuffer } },
                            { binding: 4, resource: aoView },
                            { binding: 5, resource: { buffer: kernelBuffer } },
                        ],
                    }),
                );

                const cp = passEncoder as GPUComputePassEncoder;
                cp.setPipeline(computePipeline);
                cp.setBindGroup(0, bg);
                cp.dispatchWorkgroups(
                    Math.ceil(halfW / 8),
                    Math.ceil(halfH / 8),
                );
            },
        });

        // ---- Pass 2: Bilateral Blur (compute) ------------------------------------

        const blurPipeline = this._blurPipeline;

        graph.addPass({
            name:           'SSAO/Blur',
            type:           PassType.Compute,
            reads:          [aoRaw],
            writes:         [aoBlurred],
            hasSideEffects: false,
            execute: (_encoder, passEncoder) => {
                if (!passEncoder) return;

                const aoInView   = ctx.resolveVirtualTexture(aoRaw);
                const aoOutView  = ctx.resolveVirtualTexture(aoBlurred);
                if (!aoInView || !aoOutView) return;

                const bg = this._blurBGCache.getOrCreate(
                    [aoInView, aoOutView],
                    () => ctx.backend.device.createBindGroup({
                        label:  'SSAO/Blur/BG',
                        layout: blurPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: aoInView },
                            { binding: 1, resource: { buffer: paramsBuffer } },
                            { binding: 2, resource: aoOutView },
                        ],
                    }),
                );

                const cp = passEncoder as GPUComputePassEncoder;
                cp.setPipeline(blurPipeline);
                cp.setBindGroup(0, bg);
                cp.dispatchWorkgroups(
                    Math.ceil(halfW / 8),
                    Math.ceil(halfH / 8),
                );
            },
        });

        // ---- Pass 3: Composite (fullscreen render) --------------------------------

        const compositePipeline = this._compositePipeline;
        const linearSampler     = this._linearSampler;

        graph.addPass({
            name:  'SSAO/Composite',
            type:  PassType.Render,
            reads: [inputColor, aoBlurred],
            colorAttachments: [{
                resourceId: outputColor,
                loadOp:     'clear',
                storeOp:    'store',
                clearColor: { r: 0, g: 0, b: 0, a: 1 },
            }],
            hasSideEffects: false,
            execute: (_encoder, passEncoder) => {
                if (!passEncoder) return;

                const hdrView = ctx.resolveVirtualTexture(inputColor);
                const aoView  = ctx.resolveVirtualTexture(aoBlurred);
                if (!hdrView || !aoView) return;

                const bg = this._compositeBGCache.getOrCreate(
                    [hdrView, aoView],
                    () => ctx.backend.device.createBindGroup({
                        label:  'SSAO/Composite/BG',
                        layout: compositePipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: hdrView },
                            { binding: 1, resource: aoView },
                            { binding: 2, resource: linearSampler },
                        ],
                    }),
                );

                const rp = passEncoder as GPURenderPassEncoder;
                rp.setPipeline(compositePipeline);
                rp.setBindGroup(0, bg);
                rp.draw(3);
            },
        });

        return outputColor;
    }

    destroy(): void {
        this._paramsBuffer?.destroy();
        this._kernelBuffer?.destroy();
        this._paramsBuffer      = null;
        this._kernelBuffer      = null;
        this._computePipeline   = null;
        this._blurPipeline      = null;
        this._compositePipeline = null;
        this._linearSampler     = null;
        this._computeBGCache.clear();
        this._blurBGCache.clear();
        this._compositeBGCache.clear();
    }
}
