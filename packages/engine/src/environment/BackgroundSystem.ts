// /src/engine/environment/BackgroundSystem.ts
//
// Renders the scene background behind all geometry.
// Supports four modes: solid color, vertical gradient, 2D texture, and cubemap.

import { Logger } from '../core/Logger';
import type { GPUBackend } from '../core/GPUBackend';
import type { ResourceManager, ResourceHandle } from '../core/ResourceManager';

import backgroundWGSL        from '../shaders/background.wgsl?raw';
import backgroundCubemapWGSL from '../shaders/background_cubemap.wgsl?raw';

// -------------------------------------------------------------------------
// Types
// -------------------------------------------------------------------------

export enum BackgroundType {
    Color    = 'color',
    Gradient = 'gradient',
    Texture  = 'texture',
    Cubemap  = 'cubemap',
}

export interface BackgroundConfig {
    type: BackgroundType;
    /** Solid color [r, g, b] in linear space. Used by 'color' type. */
    color?: [number, number, number];
    /** Top color for 'gradient' type. */
    topColor?: [number, number, number];
    /** Bottom color for 'gradient' type. */
    bottomColor?: [number, number, number];
    /** ResourceHandle of a 2D texture. Used by 'texture' type. */
    textureHandle?: ResourceHandle;
    /** Source path of the 2D texture (for editor preview). */
    texturePath?: string;
    /** ResourceHandle of a cube texture. Used by 'cubemap' type. */
    cubemapHandle?: ResourceHandle;
    /** Source base path of the cubemap folder (for editor preview). */
    cubemapBasePath?: string;
    /** Cubemap face file extension, e.g. '.jpg' (for editor preview). */
    cubemapExt?: string;
}

// Uniform buffer layout:
//   type        : u32     @ 0   (0=color, 1=gradient, 2=texture, 3=cubemap)
//   _pad0       : u32     @ 4
//   _pad1       : u32     @ 8
//   _pad2       : u32     @ 12
//   color       : vec4f   @ 16  (solid color or bottom color)
//   topColor    : vec4f   @ 32  (gradient top color)
//   invViewProj : mat4x4f @ 48  (inverse VP for cubemap ray direction)
//   Total = 112 bytes → pad to 128 (16-byte aligned)
const BG_UNIFORM_SIZE = 128;

// -------------------------------------------------------------------------
// BackgroundSystem
// -------------------------------------------------------------------------

export class BackgroundSystem {

    private readonly _log = new Logger('BackgroundSystem');

    private _backend!:  GPUBackend;
    private _resources!: ResourceManager;

    // --- configuration ---
    private _config: BackgroundConfig = { type: BackgroundType.Color, color: [0, 0, 0] };

    // --- GPU resources ---
    private _pipeline:          GPURenderPipeline | null = null;
    private _cubemapPipeline:   GPURenderPipeline | null = null;
    private _uniformBuffer:     GPUBuffer | null = null;
    private _bindGroupLayout:   GPUBindGroupLayout | null = null;
    private _cbBindGroupLayout: GPUBindGroupLayout | null = null;
    private _sampler:           GPUSampler | null = null;

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    init(backend: GPUBackend, resources: ResourceManager): void {
        this._backend   = backend;
        this._resources = resources;

        // Uniform buffer
        this._uniformBuffer = backend.device.createBuffer({
            label: 'Background/Uniforms',
            size:  BG_UNIFORM_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Linear clamp sampler for both 2D texture and cubemap
        this._sampler = backend.device.createSampler({
            label:        'Background/Sampler',
            magFilter:    'linear',
            minFilter:    'linear',
            mipmapFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            addressModeW: 'clamp-to-edge',
        });

        // BGL for color/gradient/texture (group 0: uniforms + tex2d + sampler)
        this._bindGroupLayout = backend.device.createBindGroupLayout({
            label: 'Background/BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ],
        });

        // BGL for cubemap (group 0: uniforms + texCube + sampler)
        this._cbBindGroupLayout = backend.device.createBindGroupLayout({
            label: 'Background/CubeBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: 'cube' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ],
        });

        // Shader modules (separate for 2D vs cube texture bindings)
        const bgModule = backend.device.createShaderModule({
            label: 'Background/Shader',
            code:  backgroundWGSL,
        });
        const cbModule = backend.device.createShaderModule({
            label: 'Background/CubemapShader',
            code:  backgroundCubemapWGSL,
        });

        const pipelineLayout = backend.device.createPipelineLayout({
            label: 'Background/PipelineLayout',
            bindGroupLayouts: [this._bindGroupLayout],
        });
        const cbPipelineLayout = backend.device.createPipelineLayout({
            label: 'Background/CubePipelineLayout',
            bindGroupLayouts: [this._cbBindGroupLayout],
        });

        const depthStencil: GPUDepthStencilState = {
            format:             'depth32float',
            depthWriteEnabled:  false,
            depthCompare:       'equal',
        };

        this._pipeline = backend.device.createRenderPipeline({
            label:  'Background/Pipeline',
            layout: pipelineLayout,
            vertex:   { module: bgModule, entryPoint: 'vs_main' },
            fragment: { module: bgModule, entryPoint: 'fs_main', targets: [{ format: 'rgba16float' }] },
            primitive: { topology: 'triangle-list' },
            depthStencil,
        });

        this._cubemapPipeline = backend.device.createRenderPipeline({
            label:  'Background/CubemapPipeline',
            layout: cbPipelineLayout,
            vertex:   { module: cbModule, entryPoint: 'vs_main' },
            fragment: { module: cbModule, entryPoint: 'fs_main', targets: [{ format: 'rgba16float' }] },
            primitive: { topology: 'triangle-list' },
            depthStencil,
        });

        this._log.info('Initialised');
    }

    destroy(): void {
        this._uniformBuffer?.destroy();
        this._uniformBuffer     = null;
        this._pipeline          = null;
        this._cubemapPipeline   = null;
        this._bindGroupLayout   = null;
        this._cbBindGroupLayout = null;
        this._sampler           = null;
    }

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    setConfig(config: BackgroundConfig): void {
        this._config = config;
    }

    getConfig(): Readonly<BackgroundConfig> {
        return this._config;
    }

    // -------------------------------------------------------------------------
    // Per-frame rendering
    // -------------------------------------------------------------------------

    /**
     * Upload uniform data and record draw commands into the given render pass.
     * @param rp              Active render pass encoder (writes to hdrColor, depth loaded).
     * @param invViewProj     Inverse view-projection matrix (Float32Array of 16 floats).
     * @param placeholder2D   A fallback 1×1 white texture handle (used when no texture is set).
     */
    render(
        rp: GPURenderPassEncoder,
        invViewProj: Float32Array,
        placeholder2D: GPUTextureView,
    ): void {
        if (!this._pipeline || !this._cubemapPipeline ||
            !this._uniformBuffer || !this._sampler ||
            !this._bindGroupLayout || !this._cbBindGroupLayout) return;

        const cfg = this._config;

        // --- Fill uniform buffer ---
        const data = new ArrayBuffer(BG_UNIFORM_SIZE);
        const u32  = new Uint32Array(data);
        const f32  = new Float32Array(data);

        // type (u32 at offset 0)
        u32[0] = cfg.type === BackgroundType.Color    ? 0
               : cfg.type === BackgroundType.Gradient  ? 1
               : cfg.type === BackgroundType.Texture   ? 2
               : /* cubemap */                           3;

        // color / bottomColor (vec4f at offset 16 → float index 4)
        const col = cfg.type === BackgroundType.Gradient ? (cfg.bottomColor ?? [0, 0, 0]) : (cfg.color ?? [0, 0, 0]);
        f32[4] = col[0]; f32[5] = col[1]; f32[6] = col[2]; f32[7] = 1.0;

        // topColor (vec4f at offset 32 → float index 8)
        const top = cfg.topColor ?? [0, 0, 0];
        f32[8] = top[0]; f32[9] = top[1]; f32[10] = top[2]; f32[11] = 1.0;

        // invViewProj (mat4x4f at offset 48 → float index 12)
        for (let i = 0; i < 16; i++) f32[12 + i] = invViewProj[i]!;

        this._backend.queue.writeBuffer(this._uniformBuffer, 0, data);

        // --- Choose pipeline and create bind group ---
        const isCubemap = cfg.type === BackgroundType.Cubemap;

        if (isCubemap) {
            // Cubemap path
            const cubeTex = cfg.cubemapHandle !== undefined
                ? this._resources.getTexture(cfg.cubemapHandle)
                : null;
            if (!cubeTex) return; // no cubemap loaded yet

            const cubeView = cubeTex.createView({
                dimension: 'cube',
                arrayLayerCount: 6,
            });

            const bg = this._backend.device.createBindGroup({
                label:  'Background/CubeBG',
                layout: this._cbBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this._uniformBuffer } },
                    { binding: 1, resource: cubeView },
                    { binding: 2, resource: this._sampler },
                ],
            });

            rp.setPipeline(this._cubemapPipeline);
            rp.setBindGroup(0, bg);
            rp.draw(3);
        } else {
            // Color / Gradient / Texture path
            let texView = placeholder2D;
            if (cfg.type === BackgroundType.Texture && cfg.textureHandle !== undefined) {
                const tex = this._resources.getTexture(cfg.textureHandle);
                if (tex) texView = tex.createView();
            }

            const bg = this._backend.device.createBindGroup({
                label:  'Background/BG',
                layout: this._bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this._uniformBuffer } },
                    { binding: 1, resource: texView },
                    { binding: 2, resource: this._sampler },
                ],
            });

            rp.setPipeline(this._pipeline);
            rp.setBindGroup(0, bg);
            rp.draw(3);
        }
    }
}
