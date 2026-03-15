// /src/engine/postprocess/PostProcessStack.ts

import type { GPUBackend } from '../core/GPUBackend';
import type { ResourceManager } from '../core/ResourceManager';
import type { RenderGraph, VirtualResourceId } from '../rendergraph/RenderGraph';
import type { ShaderSystem } from '../shaders/ShaderSystem';
import type { PipelineManager } from '../pipelines/PipelineManager';
// -------------------------------------------------------------------------
// Post-process context — everything effects need to create GPU work
// -------------------------------------------------------------------------

/**
 * Passed to every effect's addPasses() so it has access to G-Buffer
 * virtual resources, GPU subsystems, and per-frame data.
 */
export interface PostProcessContext {
    readonly backend:         GPUBackend;
    readonly resources:       ResourceManager;
    readonly shaderSystem:    ShaderSystem;
    readonly pipelineManager: PipelineManager;

    /** Virtual resource IDs for G-Buffer textures (set each frame). */
    readonly gbAlbedoAO:         VirtualResourceId;
    readonly gbNormalRoughness:  VirtualResourceId;
    readonly gbMetallicEmissive: VirtualResourceId;

    /** Per-frame uniform buffer (current ring slot). */
    readonly perFrameBuffer:     GPUBuffer;
    /** Nearest-clamp sampler shared across passes. */
    readonly clampSampler:       GPUSampler;

    /** Current frame index (for ring-buffer slot selection). */
    readonly frameIndex:         number;
    /** Canvas resolution [width, height]. */
    readonly resolution:         [number, number];

    /** Resolve a virtual resource id to a GPUTextureView (only valid inside execute callbacks). */
    resolveVirtualTexture(id: VirtualResourceId, viewDesc?: GPUTextureViewDescriptor): GPUTextureView | undefined;

    /** Environment cubemap texture (6-layer). Null/undefined if no cubemap is loaded. */
    readonly envCubemapTexture?: GPUTexture | null;
    /** Whether the env cubemap is active (not just a placeholder). */
    readonly envCubemapEnabled?: boolean;
    /** Linear sampler for cubemap sampling. */
    readonly envCubemapSampler?: GPUSampler | null;
}

// -------------------------------------------------------------------------
// Post-process effect interface
// -------------------------------------------------------------------------

export type PostEffectHandle = number;

/**
 * Every post-process effect implements this interface so it can be
 * composed into the stack and wired into the render graph.
 */
export interface PostProcessEffect {
    /** Unique name used for debugging and render-graph pass labels. */
    readonly name: string;
    /** Whether this effect is currently active. */
    enabled: boolean;

    /**
     * Called once at init to create any persistent GPU resources
     * (e.g. blur kernels, LUT textures, pipelines).
     */
    init(ctx: PostProcessContext): void;

    /**
     * Register this effect's pass(es) into the render graph.
     * @param graph        The render graph to add passes to.
     * @param ctx          Post-process context with G-Buffer resources and GPU subsystems.
     * @param inputColor   Virtual resource id of the HDR colour input.
     * @param inputDepth   Virtual resource id of the full-resolution depth buffer.
     * @param halfDepth    Virtual resource id of the half-resolution r32float depth.
     * @returns            Virtual resource id of the output colour (may be the same as input).
     */
    addPasses(
        graph:      RenderGraph,
        ctx:        PostProcessContext,
        inputColor: VirtualResourceId,
        inputDepth: VirtualResourceId,
        halfDepth?: VirtualResourceId,
    ): VirtualResourceId;

    /** Release GPU resources. */
    destroy(): void;
}

// -------------------------------------------------------------------------
// Built-in effect stubs
// -------------------------------------------------------------------------

/**
 * Physically-based bloom (downsample–upsample chain).
 */
export class BloomEffect implements PostProcessEffect {
    readonly name = 'Bloom';
    enabled = false; // disabled until implemented
    threshold: number = 1.0;
    intensity: number = 0.04;
    mipLevels: number = 6;

    init(_ctx: PostProcessContext): void {
        // TODO: nothing persistent; mip chain is transient
    }

    addPasses(_graph: RenderGraph, _ctx: PostProcessContext, inputColor: VirtualResourceId, _inputDepth?: VirtualResourceId, _halfDepth?: VirtualResourceId): VirtualResourceId {
        return inputColor; // placeholder
    }

    destroy(): void { /* TODO */ }
}

/**
 * Tonemapping (ACES / AgX / Khronos PBR Neutral).
 */
export class TonemapEffect implements PostProcessEffect {
    readonly name = 'Tonemap';
    enabled = false; // disabled until implemented
    operator: 'ACES' | 'AgX' | 'Reinhard' | 'PBR_Neutral' = 'ACES';

    init(_ctx: PostProcessContext): void {
        // TODO: load LUT if using AgX
    }

    addPasses(_graph: RenderGraph, _ctx: PostProcessContext, inputColor: VirtualResourceId, _inputDepth?: VirtualResourceId, _halfDepth?: VirtualResourceId): VirtualResourceId {
        return inputColor; // placeholder
    }

    destroy(): void { /* TODO */ }
}

/**
 * Temporal Anti-Aliasing.
 */
export class TAAEffect implements PostProcessEffect {
    readonly name = 'TAA';
    enabled = false; // disabled until implemented

    init(_ctx: PostProcessContext): void {
        // TODO: create history buffer (persistent across frames)
    }

    addPasses(_graph: RenderGraph, _ctx: PostProcessContext, inputColor: VirtualResourceId, _inputDepth?: VirtualResourceId, _halfDepth?: VirtualResourceId): VirtualResourceId {
        return inputColor; // placeholder
    }

    destroy(): void { /* TODO */ }
}

/**
 * Simple FXAA as a lightweight fallback to TAA.
 */
export class FXAAEffect implements PostProcessEffect {
    readonly name = 'FXAA';
    enabled = false; // off by default

    init(_ctx: PostProcessContext): void { /* TODO */ }

    addPasses(_graph: RenderGraph, _ctx: PostProcessContext, inputColor: VirtualResourceId, _inputDepth?: VirtualResourceId, _halfDepth?: VirtualResourceId): VirtualResourceId {
        return inputColor;
    }

    destroy(): void { /* TODO */ }
}

// ─────────────────────────────────────────────────────────────────────────────
// Depth-buffer effects
// ─────────────────────────────────────────────────────────────────────────────

// SSAO is implemented in its own file: SSAOEffect.ts
import { SSAOEffect } from './SSAOEffect';
export { SSAOEffect } from './SSAOEffect';

/**
 * Volumetric Fog.
 */
export class VolumetricFogEffect implements PostProcessEffect {
    readonly name = 'VolumetricFog';
    enabled = false; // disabled until implemented

    density:    number = 0.02;
    scattering: number = 0.5;
    absorption: number = 0.01;
    steps:      number = 32;

    init(_ctx: PostProcessContext): void {
        // TODO: create noise volume texture (3D Perlin/Worley) for density variation
    }

    addPasses(_graph: RenderGraph, _ctx: PostProcessContext, inputColor: VirtualResourceId, _inputDepth?: VirtualResourceId, _halfDepth?: VirtualResourceId): VirtualResourceId {
        return inputColor; // passthrough until implemented
    }

    destroy(): void { /* TODO */ }
}

/**
 * Contact Shadows.
 */
export class ContactShadowsEffect implements PostProcessEffect {
    readonly name = 'ContactShadows';
    enabled = false; // disabled until implemented

    rayLength:  number = 0.5;
    steps:      number = 16;
    softness:   number = 0.2;

    init(_ctx: PostProcessContext): void {
        // TODO: no persistent resources needed
    }

    addPasses(_graph: RenderGraph, _ctx: PostProcessContext, inputColor: VirtualResourceId, _inputDepth?: VirtualResourceId, _halfDepth?: VirtualResourceId): VirtualResourceId {
        return inputColor; // passthrough until implemented
    }

    destroy(): void { /* TODO */ }
}

// -------------------------------------------------------------------------
// Stack
// -------------------------------------------------------------------------

/**
 * Ordered chain of post-process effects.
 *
 * Each effect receives the output of the previous one and produces
 * a new (or the same) virtual colour resource.  The stack is fully
 * composable — effects can be reordered, toggled, or removed at runtime.
 */
export class PostProcessStack {

    private _ctx!: PostProcessContext;
    private _effects: PostProcessEffect[] = [];

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    init(ctx: PostProcessContext): void {
        this._ctx = ctx;

        // Default chain order.
        this._effects = [
            new SSAOEffect(),
            new VolumetricFogEffect(),
            new ContactShadowsEffect(),
            new BloomEffect(),
            new TonemapEffect(),
            new TAAEffect(),
            new FXAAEffect(),
        ];

        for (const effect of this._effects) {
            effect.init(ctx);
        }
    }

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    insertEffect(effect: PostProcessEffect, index: number): void {
        effect.init(this._ctx);
        this._effects.splice(index, 0, effect);
    }

    removeEffect(name: string): void {
        const idx = this._effects.findIndex(e => e.name === name);
        if (idx >= 0) {
            this._effects[idx]!.destroy();
            this._effects.splice(idx, 1);
        }
    }

    getEffect<T extends PostProcessEffect>(name: string): T | undefined {
        return this._effects.find(e => e.name === name) as T | undefined;
    }

    setEnabled(name: string, enabled: boolean): void {
        const effect = this._effects.find(e => e.name === name);
        if (effect) effect.enabled = enabled;
    }

    // -------------------------------------------------------------------------
    // Render-graph integration
    // -------------------------------------------------------------------------

    /**
     * Walk the effect chain, adding each enabled effect's passes to the graph.
     * Returns the final colour virtual resource id to blit to the swap chain.
     *
     * @param graph      The current frame's render graph.
     * @param ctx        Post-process context for this frame.
     * @param hdrColor   Virtual resource id of the HDR colour output from the lighting pass.
     * @param depth      Virtual resource id of the full-resolution depth buffer.
     * @param halfDepth  Virtual resource id of the half-resolution r32float depth.
     */
    addPasses(
        graph:     RenderGraph,
        ctx:       PostProcessContext,
        hdrColor:  VirtualResourceId,
        depth:     VirtualResourceId,
        halfDepth?: VirtualResourceId,
    ): VirtualResourceId {
        let currentColor = hdrColor;

        for (const effect of this._effects) {
            if (!effect.enabled) continue;
            currentColor = effect.addPasses(graph, ctx, currentColor, depth, halfDepth);
        }

        return currentColor;
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    destroy(): void {
        for (const effect of this._effects) {
            effect.destroy();
        }
        this._effects.length = 0;
    }
}
