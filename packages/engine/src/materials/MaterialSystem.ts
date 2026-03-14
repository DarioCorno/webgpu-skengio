// /src/engine/materials/MaterialSystem.ts

import type { GPUBackend } from '../core/GPUBackend';
import type { ResourceManager, ResourceHandle } from '../core/ResourceManager';
import type { ShaderSystem, ShaderDefines } from '../shaders/ShaderSystem';
import type { EngineConfiguration } from '../core/EngineConfiguration';

// -------------------------------------------------------------------------
// Enums & constants
// -------------------------------------------------------------------------

/**
 * The rendering path a material will be routed to.
 * This is the critical deferred-vs-forward classification.
 */
export enum RenderPath {
    /** Rendered into G-Buffer during the deferred geometry pass. */
    Deferred = 'DEFERRED',
    /** Rendered in the forward transparency pass (sorted back-to-front). */
    Forward  = 'FORWARD',
}

export enum AlphaMode {
    Opaque    = 'OPAQUE',
    Mask      = 'MASK',      // alpha-test (deferred-compatible)
    Blend     = 'BLEND',     // requires forward pass
}

export enum MaterialShadingModel {
    PBR_MetallicRoughness  = 'PBR_MR',
    PBR_SpecularGlossiness = 'PBR_SG',
    Unlit = 'UNLIT',
}

// -------------------------------------------------------------------------
// Material descriptor
// -------------------------------------------------------------------------

export type MaterialHandle = number;

/**
 * PBR material parameters (metallic-roughness workflow).
 */
export interface PBRParameters {
    baseColorFactor:   [number, number, number, number]; // RGBA
    metallicFactor:    number;
    roughnessFactor:   number;
    emissiveFactor:    [number, number, number];
    normalScale:       number;
    occlusionStrength: number;
    alphaCutoff:       number; // used when alphaMode == Mask
    /** Overall surface opacity [0,1]. 1.0 = fully opaque. Used when alphaMode == Blend. */
    opacity:           number;
    /** Index of refraction for Fresnel. 1.5 = glass. Used when alphaMode == Blend. */
    ior:               number;
    /** Shadow opacity [0,1]. 0 = no shadow, 1 = full shadow. Default: 1. */
    shadowOpacity:     number;
}

/**
 * Texture slots a PBR material can reference.
 */
export interface MaterialTextures {
    baseColorMap?:         ResourceHandle;
    normalMap?:            ResourceHandle;
    metallicRoughnessMap?: ResourceHandle;
    occlusionMap?:         ResourceHandle;
    emissiveMap?:          ResourceHandle;
}

export interface MaterialDescriptor {
    label?:        string;
    shadingModel?: MaterialShadingModel;
    alphaMode?:    AlphaMode;
    doubleSided?:  boolean;
    pbrParams?:    Partial<PBRParameters>;
    textures?:     MaterialTextures;
    /** Whether this material casts shadows. Default: true. */
    castShadow?: boolean;
}

// -------------------------------------------------------------------------
// Uniform buffer layout (must match the WGSL MaterialUniforms struct)
// -------------------------------------------------------------------------
//
//   baseColorFactor    : vec4f  = 16 bytes  @ offset  0
//   metallicFactor     : f32    =  4 bytes  @ offset 16
//   roughnessFactor    : f32    =  4 bytes  @ offset 20
//   normalScale        : f32    =  4 bytes  @ offset 24
//   occlusionStrength  : f32    =  4 bytes  @ offset 28
//   emissiveFactor     : vec3f  = 12 bytes  @ offset 32  (align 16)
//   alphaCutoff        : f32    =  4 bytes  @ offset 44
//   opacity            : f32    =  4 bytes  @ offset 48
//   ior                : f32    =  4 bytes  @ offset 52
//   shadowOpacity      : f32    =  4 bytes  @ offset 56
//   _pad               : f32    =  4 bytes  @ offset 60  (pad to 64)
//                                 ——————————
//   Total                       = 64 bytes  (4 × 16, 16-byte aligned)
//
const MATERIAL_UNIFORM_SIZE = 64; // bytes

/**
 * Internal record stored per material.
 */
export interface MaterialRecord {
    handle:        MaterialHandle;
    label:         string;
    renderPath:    RenderPath;
    alphaMode:     AlphaMode;
    shadingModel:  MaterialShadingModel;
    doubleSided:   boolean;
    pbrParams:     PBRParameters;
    textures:      MaterialTextures;
    /** Whether this material casts shadows. */
    castShadow: boolean;
    /** GPU uniform buffer holding packed PBR params. */
    uniformBuffer: ResourceHandle;
    /** Shader preprocessor defines for variant selection. */
    shaderDefines: ShaderDefines;
    dirty:         boolean;
    /**
     * Lazily-created bind group for @group(1) in the G-Buffer pipeline.
     * Created on first use via getOrCreateBindGroup(); invalidated when the
     * material is destroyed.
     */
    bindGroup?: GPUBindGroup;
}

/**
 * Owns all materials and their GPU resources (uniform buffers, bind groups).
 *
 * Key responsibility: classify each material into Deferred (opaque) or
 * Forward (transparent) render path, and produce the correct shader
 * defines so the ShaderSystem / PipelineManager compiles the right variant.
 */
export class MaterialSystem {

    private _backend!: GPUBackend;
    private _resources!: ResourceManager;
    private _shaderSystem!: ShaderSystem;
    private _nextHandle: MaterialHandle = 1;
    private _materials: Map<MaterialHandle, MaterialRecord> = new Map();

    // Lazily-created shared sampler used by all textured materials.
    // Linear filtering, repeat wrap — suitable for most PBR maps.
    // Anisotropy level is read from the engine configuration.
    private _defaultSampler: ResourceHandle | undefined;
    private _maxAnisotropy: number = 4;

    // Default PBR values (white, fully metallic/rough, no emissive)
    static readonly DEFAULT_PBR: PBRParameters = {
        baseColorFactor:   [1, 1, 1, 1],
        metallicFactor:    0.0,
        roughnessFactor:   0.5,
        emissiveFactor:    [0, 0, 0],
        normalScale:       1.0,
        occlusionStrength: 1.0,
        alphaCutoff:       0.5,
        opacity:           1.0,
        ior:               1.5,
        shadowOpacity:     1.0,
    };

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    init(backend: GPUBackend, resources: ResourceManager, shaderSystem: ShaderSystem, config?: EngineConfiguration): void {
        this._backend        = backend;
        this._resources      = resources;
        this._shaderSystem   = shaderSystem;
        this._maxAnisotropy  = config?.maxAnisotropy ?? 4;
        this._defaultSampler = undefined; // force re-creation with new anisotropy
    }

    // -------------------------------------------------------------------------
    // Material CRUD
    // -------------------------------------------------------------------------

    createMaterial(desc: MaterialDescriptor): MaterialHandle {
        const handle       = this._nextHandle++;
        const alphaMode    = desc.alphaMode   ?? AlphaMode.Opaque;
        const renderPath   = this._classifyRenderPath(alphaMode);
        const shadingModel = desc.shadingModel ?? MaterialShadingModel.PBR_MetallicRoughness;
        const shaderDefines = this._buildDefines(desc);

        // Merge caller-supplied params over defaults
        const pbrParams: PBRParameters = {
            ...MaterialSystem.DEFAULT_PBR,
            ...desc.pbrParams,
        };

        // Allocate the GPU uniform buffer (UNIFORM | COPY_DST so writeBuffer can update it)
        const uniformBuffer = this._resources.createBuffer({
            label: `${desc.label ?? 'material'}_${handle}_ub`,
            size:  MATERIAL_UNIFORM_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const record: MaterialRecord = {
            handle,
            label:         desc.label    ?? `material_${handle}`,
            renderPath,
            alphaMode,
            shadingModel,
            doubleSided:   desc.doubleSided ?? false,
            castShadow:    desc.castShadow ?? true,
            pbrParams,
            textures:      { ...desc.textures },
            uniformBuffer,
            shaderDefines,
            dirty:         true, // needs initial upload
        };

        this._materials.set(handle, record);

        // Pack params into the GPU buffer immediately
        this._uploadMaterial(record);
        record.dirty = false;

        return handle;
    }

    getMaterial(handle: MaterialHandle): MaterialRecord | undefined {
        return this._materials.get(handle);
    }

    /** Iterate all registered materials. */
    getMaterials(): IterableIterator<MaterialRecord> {
        return this._materials.values();
    }

    updateMaterial(handle: MaterialHandle, changes: Partial<MaterialDescriptor>): void {
        const mat = this._materials.get(handle);
        if (!mat) return;

        if (changes.pbrParams)    Object.assign(mat.pbrParams, changes.pbrParams);
        if (changes.alphaMode !== undefined) {
            mat.alphaMode  = changes.alphaMode;
            mat.renderPath = this._classifyRenderPath(changes.alphaMode);
        }
        if (changes.doubleSided !== undefined) mat.doubleSided = changes.doubleSided;
        if (changes.textures) {
            Object.assign(mat.textures, changes.textures);
            mat.bindGroup = undefined; // texture slots changed → must rebuild bind group
        }
        if (changes.shadingModel !== undefined) mat.shadingModel = changes.shadingModel;
        if (changes.castShadow !== undefined) mat.castShadow = changes.castShadow;

        // Rebuild shader defines after any change
        mat.shaderDefines = this._buildDefines({
            alphaMode:    mat.alphaMode,
            doubleSided:  mat.doubleSided,
            textures:     mat.textures,
            shadingModel: mat.shadingModel,
        });

        mat.dirty = true;
    }

    /**
     * Return (or lazily create) the GPUBindGroup for @group(1) in the G-Buffer
     * pipeline.  The bind group wraps only the material uniform buffer.
     *
     * @param bgl  The GPUBindGroupLayout obtained from
     *             `pipeline.getBindGroupLayout(1)` for the target pipeline.
     *             Must remain compatible across calls (single-pipeline assumption).
     */
    getOrCreateBindGroup(handle: MaterialHandle, bgl: GPUBindGroupLayout): GPUBindGroup | undefined {
        const mat = this._materials.get(handle);
        if (!mat) return undefined;

        // Always refresh lastUsedFrame on the uniform buffer.
        const gpuBuffer = this._resources.getBuffer(mat.uniformBuffer);
        if (!gpuBuffer) return undefined;

        if (!mat.bindGroup) {
            const entries: GPUBindGroupEntry[] = [
                { binding: 0, resource: { buffer: gpuBuffer } },
            ];

            const tex = mat.textures;
            const hasTextures = !!(
                tex.baseColorMap !== undefined || tex.normalMap !== undefined ||
                tex.metallicRoughnessMap !== undefined || tex.occlusionMap !== undefined ||
                tex.emissiveMap !== undefined
            );

            if (hasTextures) {
                // Binding 1 — shared sampler (present whenever any texture map is active).
                //
                // TODO (Megatexture): Move this sampler to @group(0) in the per-frame
                //   bind group so it is shared across all draw calls at zero per-material cost.
                const sampler = this._resources.getSampler(this._getDefaultSampler());
                if (sampler) entries.push({ binding: 1, resource: sampler });

                // Helper: resolve handle → GPUTextureView and push an entry.
                const addTex = (binding: number, h: ResourceHandle | undefined) => {
                    if (h === undefined) return;
                    const view = this._resources.createView(h);
                    if (view) entries.push({ binding, resource: view });
                };

                // Bindings 2–6 match the @group(1) layout in gbuffer.wgsl.
                addTex(2, tex.baseColorMap);
                addTex(3, tex.normalMap);
                addTex(4, tex.metallicRoughnessMap);
                addTex(5, tex.occlusionMap);
                addTex(6, tex.emissiveMap);
            }

            mat.bindGroup = this._backend.device.createBindGroup({
                label:   `${mat.label}_bg`,
                layout:  bgl,
                entries,
            });
        }

        return mat.bindGroup;
    }

    /**
     * Create a non-cached bind group for the shadow transparent pipeline.
     * Only includes the uniform buffer, sampler, and base color map (bindings 0–2).
     * Does NOT cache on the material record so it won't conflict with the
     * forward/G-Buffer pipeline's cached bind group.
     */
    createShadowBindGroup(handle: MaterialHandle, bgl: GPUBindGroupLayout): GPUBindGroup | undefined {
        const mat = this._materials.get(handle);
        if (!mat) return undefined;
        const gpuBuffer = this._resources.getBuffer(mat.uniformBuffer);
        if (!gpuBuffer) return undefined;

        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: gpuBuffer } },
        ];

        const tex = mat.textures;
        const hasTextures = tex.baseColorMap !== undefined;

        if (hasTextures) {
            const sampler = this._resources.getSampler(this._getDefaultSampler());
            if (sampler) entries.push({ binding: 1, resource: sampler });
            if (tex.baseColorMap !== undefined) {
                const view = this._resources.createView(tex.baseColorMap);
                if (view) entries.push({ binding: 2, resource: view });
            }
        }

        return this._backend.device.createBindGroup({
            label:   `${mat.label}_shadow_bg`,
            layout:  bgl,
            entries,
        });
    }

    // -------------------------------------------------------------------------
    // Default sampler
    // -------------------------------------------------------------------------

    /**
     * Lazily create (and cache) the shared linear-repeat sampler used by all
     * textured materials.  4× anisotropy is a reasonable default for PBR maps.
     */
    private _getDefaultSampler(): ResourceHandle {
        if (this._defaultSampler === undefined) {
            this._defaultSampler = this._resources.createSampler({
                label:        'material_default_sampler',
                addressModeU: 'repeat',
                addressModeV: 'repeat',
                magFilter:    'linear',
                minFilter:    'linear',
                mipmapFilter: 'linear',
                maxAnisotropy: this._maxAnisotropy,
            });
        }
        return this._defaultSampler;
    }

    destroyMaterial(handle: MaterialHandle): void {
        const mat = this._materials.get(handle);
        if (!mat) return;
        this._resources.destroyBuffer(mat.uniformBuffer);
        // GPUBindGroup has no destroy() — just drop the reference.
        this._materials.delete(handle);
    }

    // -------------------------------------------------------------------------
    // Render-path classification
    // -------------------------------------------------------------------------

    getRenderPath(handle: MaterialHandle): RenderPath {
        return this._materials.get(handle)?.renderPath ?? RenderPath.Deferred;
    }

    /**
     * Opaque and alpha-masked materials go through deferred.
     * Alpha-blended materials require the forward path.
     */
    private _classifyRenderPath(alphaMode: AlphaMode): RenderPath {
        return alphaMode === AlphaMode.Blend ? RenderPath.Forward : RenderPath.Deferred;
    }

    // -------------------------------------------------------------------------
    // Shader defines
    // -------------------------------------------------------------------------

    /**
     * Produce the set of shader preprocessor defines for this material.
     * Used by PipelineManager to select the correct shader variant.
     */
    private _buildDefines(desc: MaterialDescriptor): ShaderDefines {
        const d: ShaderDefines = {};

        const t = desc.textures;
        const hasTextures = !!(t && (
            t.baseColorMap !== undefined || t.normalMap !== undefined ||
            t.metallicRoughnessMap !== undefined || t.occlusionMap !== undefined ||
            t.emissiveMap !== undefined
        ));

        if (hasTextures)                         d['HAS_TEXTURES']       = '1';
        if (t?.baseColorMap        !== undefined) d['HAS_BASE_COLOR_MAP'] = '1';
        if (t?.normalMap           !== undefined) d['HAS_NORMAL_MAP']     = '1';
        if (t?.metallicRoughnessMap !== undefined) d['HAS_MR_MAP']        = '1';
        if (t?.occlusionMap        !== undefined) d['HAS_AO_MAP']         = '1';
        if (t?.emissiveMap         !== undefined) d['HAS_EMISSIVE_MAP']   = '1';

        if (desc.alphaMode === AlphaMode.Mask) d['ALPHA_MASK']   = '1';
        if (desc.doubleSided)                  d['DOUBLE_SIDED'] = '1';

        if (desc.shadingModel === MaterialShadingModel.Unlit) d['UNLIT'] = '1';

        return d;
    }

    // -------------------------------------------------------------------------
    // Per-frame upload
    // -------------------------------------------------------------------------

    /**
     * Upload all dirty material uniform buffers to the GPU.
     * Called once per frame before rendering.
     */
    uploadDirtyMaterials(): void {
        for (const mat of this._materials.values()) {
            if (mat.dirty) {
                this._uploadMaterial(mat);
                mat.dirty = false;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * Pack PBRParameters into a Float32Array and write to the material's
     * uniform buffer. Layout matches MATERIAL_UNIFORM_SIZE = 48 bytes.
     */
    private _uploadMaterial(mat: MaterialRecord): void {
        const p    = mat.pbrParams;
        const data = new Float32Array(MATERIAL_UNIFORM_SIZE / 4); // 16 floats

        // vec4  baseColorFactor  @ floats 0–3
        data[0] = p.baseColorFactor[0];
        data[1] = p.baseColorFactor[1];
        data[2] = p.baseColorFactor[2];
        data[3] = p.baseColorFactor[3];

        // f32   metallicFactor   @ float 4
        data[4] = p.metallicFactor;
        // f32   roughnessFactor  @ float 5
        data[5] = p.roughnessFactor;
        // f32   normalScale      @ float 6
        data[6] = p.normalScale;
        // f32   occlusionStrength@ float 7
        data[7] = p.occlusionStrength;

        // vec3  emissiveFactor   @ floats 8–10  (offset 32 bytes, align 16 ✓)
        data[8]  = p.emissiveFactor[0];
        data[9]  = p.emissiveFactor[1];
        data[10] = p.emissiveFactor[2];

        // f32   alphaCutoff      @ float 11  (offset 44 bytes)
        data[11] = p.alphaCutoff;

        // f32   opacity          @ float 12  (offset 48 bytes)
        data[12] = p.opacity;
        // f32   ior              @ float 13  (offset 52 bytes)
        data[13] = p.ior;
        // f32   shadowOpacity   @ float 14  (offset 56 bytes)
        data[14] = p.shadowOpacity;
        // float 15 is padding (offset 60–63)

        this._resources.writeBuffer(mat.uniformBuffer, data);
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    destroy(): void {
        for (const handle of this._materials.keys()) {
            this.destroyMaterial(handle);
        }
        if (this._defaultSampler !== undefined) {
            this._resources.destroySampler(this._defaultSampler);
            this._defaultSampler = undefined;
        }
    }
}
