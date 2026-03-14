// /src/engine/core/ResourceManager.ts

import { Logger } from './Logger';
import type { GPUBackend } from './GPUBackend';

// -------------------------------------------------------------------------
// Resource handle types
// -------------------------------------------------------------------------

export type ResourceHandle = number; // opaque integer id

export interface BufferDescriptor {
    label?: string;
    size: number;
    usage: GPUBufferUsageFlags;
    mappedAtCreation?: boolean;
}

export interface TextureDescriptor {
    label?: string;
    size: GPUExtent3DStrict;
    format: GPUTextureFormat;
    usage: GPUTextureUsageFlags;
    mipLevelCount?: number;
    sampleCount?: number;
    dimension?: GPUTextureDimension;
}

export interface SamplerDescriptor {
    label?: string;
    addressModeU?: GPUAddressMode;
    addressModeV?: GPUAddressMode;
    magFilter?: GPUFilterMode;
    minFilter?: GPUFilterMode;
    mipmapFilter?: GPUMipmapFilterMode;
    maxAnisotropy?: number;
    compare?: GPUCompareFunction;
}

export interface LoadTextureOptions {
    /** Debug label for the GPU texture. Defaults to the last path segment of the URL. */
    label?: string;
    /**
     * GPU texture format. Default: 'rgba8unorm'.
     * Use 'rgba8unorm-srgb' when the image stores sRGB colour data and your
     * pipeline reads linear colour (the GPU applies gamma expand automatically).
     */
    format?: GPUTextureFormat;
    /**
     * Flip the image vertically before uploading. Default: false.
     * WebGPU UV origin is top-left; most image formats also start at top-left,
     * so flipY is rarely needed.
     */
    flipY?: boolean;
    /**
     * Colour space the image data is encoded in. Default: 'srgb'.
     * Passed to copyExternalImageToTexture so the driver can apply
     * the correct transfer function.
     */
    colorSpace?: PredefinedColorSpace;

    // TODO (Megatexture): Add an `atlasHint?: boolean` option here.
    //   When true, loadImageToTexture will route the upload through a
    //   MegatextureAllocator that packs multiple images into a single large
    //   GPUTexture atlas.  The returned ResourceHandle will refer to the
    //   atlas texture, and the associated UV offset/scale will be stored
    //   alongside the handle so MaterialSystem can write it into the
    //   MaterialUniforms buffer.  Individual per-texture GPUTexture objects
    //   will no longer be created when the atlas path is taken.
}

// -------------------------------------------------------------------------
// Internal tracked resource types
// -------------------------------------------------------------------------

interface TrackedBuffer {
    resource: GPUBuffer;
    label: string;
    lastUsedFrame: number;
}

interface TrackedTexture {
    resource: GPUTexture;
    label: string;
    lastUsedFrame: number;
    /** true → belongs to the transient pool when not in use */
    transient: boolean;
    /** Pre-computed pool key for returning to pool without re-hashing. */
    poolKey: string;
}

interface TrackedSampler {
    resource: GPUSampler;
    label: string;
}

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------


// -------------------------------------------------------------------------
// ResourceManager
// -------------------------------------------------------------------------

/**
 * Centralized ownership of all GPU resources.
 *
 * Responsibilities:
 *  - Create / cache / destroy buffers, textures, samplers, bind groups
 *  - Pool transient textures (G-Buffer targets, shadow maps, temp RT):
 *      acquireTransientTexture() → use → releaseTransientTexture()
 *      At beginFrame() any unreleased transient textures are auto-returned.
 *  - Bind-group-layout deduplication via stable JSON hash.
 */
export class ResourceManager {

    private _backend!: GPUBackend;
    private _nextHandle: ResourceHandle = 1;
    private readonly _log = new Logger('ResourceManager');

    // --- resource registries --------------------------------------------------
    private _buffers:  Map<ResourceHandle, TrackedBuffer>  = new Map();
    private _textures: Map<ResourceHandle, TrackedTexture> = new Map();
    private _samplers: Map<ResourceHandle, TrackedSampler> = new Map();

    // --- bind group layout dedup cache ----------------------------------------
    private _bindGroupLayouts: Map<string, GPUBindGroupLayout> = new Map();

    // --- transient texture pool -----------------------------------------------
    // poolKey → stack of available GPUTexture instances
    private _texturePool: Map<string, GPUTexture[]> = new Map();

    private _currentFrame: number = 0;

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    init(backend: GPUBackend): void {
        this._backend = backend;
    }

    /**
     * Call once per frame.
     * Returns all still-tracked transient textures to the pool
     * (handles the case where the render graph forgot to release them).
     * Persistent resources are only destroyed via explicit destroyBuffer()/destroyTexture() calls.
     */
    beginFrame(frameIndex: number): void {
        this._currentFrame = frameIndex;

        // --- auto-return unreleased transient textures -----------------------
        for (const [handle, tracked] of this._textures) {
            if (tracked.transient) {
                this._returnToPool(tracked);
                this._textures.delete(handle);
            }
        }

    }

    // -------------------------------------------------------------------------
    // Buffers
    // -------------------------------------------------------------------------

    createBuffer(desc: BufferDescriptor): ResourceHandle {
        const label  = desc.label ?? `buffer_${this._nextHandle}`;
        const buffer = this._backend.device.createBuffer({
            label,
            size:             desc.size,
            usage:            desc.usage,
            mappedAtCreation: desc.mappedAtCreation ?? false,
        });

        const handle = this._nextHandle++;
        this._buffers.set(handle, { resource: buffer, label, lastUsedFrame: this._currentFrame });
        this._log.debug(`Created buffer "${label}" size=${desc.size} handle=${handle}`);
        return handle;
    }

    getBuffer(handle: ResourceHandle): GPUBuffer | undefined {
        const tracked = this._buffers.get(handle);
        if (tracked) tracked.lastUsedFrame = this._currentFrame;
        return tracked?.resource;
    }

    /**
     * Write CPU data into a GPU buffer via queue.writeBuffer.
     * @param offset  Byte offset into the buffer. Defaults to 0.
     */
    writeBuffer(handle: ResourceHandle, data: GPUAllowSharedBufferSource, offset: number = 0): void {
        const buffer = this.getBuffer(handle);
        if (!buffer) {
            this._log.warn(`writeBuffer: handle ${handle} not found`);
            return;
        }
        this._backend.queue.writeBuffer(buffer, offset, data);
    }

    destroyBuffer(handle: ResourceHandle): void {
        const tracked = this._buffers.get(handle);
        if (!tracked) return;
        tracked.resource.destroy();
        this._buffers.delete(handle);
        this._log.debug(`Destroyed buffer "${tracked.label}" handle=${handle}`);
    }

    // -------------------------------------------------------------------------
    // Textures
    // -------------------------------------------------------------------------

    createTexture(desc: TextureDescriptor, transient: boolean = false): ResourceHandle {
        const poolKey = _texturePoolKey(desc);

        // For transient textures, try the pool first
        if (transient) {
            const pool = this._texturePool.get(poolKey);
            if (pool && pool.length > 0) {
                const recycled = pool.pop()!;
                const handle   = this._nextHandle++;
                this._textures.set(handle, {
                    resource: recycled,
                    label:    recycled.label ?? poolKey,
                    lastUsedFrame: this._currentFrame,
                    transient: true,
                    poolKey,
                });
                this._log.debug(`Recycled transient texture "${poolKey}" handle=${handle}`);
                return handle;
            }
        }

        const label   = desc.label ?? `texture_${this._nextHandle}`;
        const texture = this._backend.device.createTexture({
            label,
            size:          desc.size,
            format:        desc.format,
            usage:         desc.usage,
            mipLevelCount: desc.mipLevelCount  ?? 1,
            sampleCount:   desc.sampleCount    ?? 1,
            dimension:     desc.dimension      ?? '2d',
        });

        const handle = this._nextHandle++;
        this._textures.set(handle, {
            resource: texture,
            label,
            lastUsedFrame: this._currentFrame,
            transient,
            poolKey,
        });
        this._log.debug(`Created ${transient ? 'transient ' : ''}texture "${label}" format=${desc.format} handle=${handle}`);
        return handle;
    }

    getTexture(handle: ResourceHandle): GPUTexture | undefined {
        const tracked = this._textures.get(handle);
        if (tracked) tracked.lastUsedFrame = this._currentFrame;
        return tracked?.resource;
    }

    /**
     * Convenience: create a GPUTextureView directly from a handle.
     * The view is not tracked — callers own its lifetime.
     */
    createView(handle: ResourceHandle, desc?: GPUTextureViewDescriptor): GPUTextureView | undefined {
        return this.getTexture(handle)?.createView(desc);
    }

    /**
     * Acquire a transient texture from the pool (or create one).
     * Must be paired with releaseTransientTexture() or it will be
     * auto-returned at the next beginFrame().
     */
    acquireTransientTexture(desc: TextureDescriptor): ResourceHandle {
        return this.createTexture(desc, true);
    }

    /**
     * Return a transient texture to the pool immediately.
     * Call this as soon as a render pass no longer needs the texture.
     */
    releaseTransientTexture(handle: ResourceHandle): void {
        const tracked = this._textures.get(handle);
        if (!tracked || !tracked.transient) return;
        this._returnToPool(tracked);
        this._textures.delete(handle);
        this._log.debug(`Released transient texture "${tracked.label}" handle=${handle} → pool`);
    }

    /**
     * Fetch an image from `url`, decode it via `createImageBitmap`, upload it
     * to a new GPU texture, and return the ResourceHandle.
     *
     * The texture is created with TEXTURE_BINDING | COPY_DST | RENDER_ATTACHMENT
     * usage so it can be sampled in shaders and used as a render target if needed
     * (required by copyExternalImageToTexture on some backends).
     *
     * Mip-map generation is NOT performed — the texture has mipLevelCount=1.
     * A full HZB-style compute mip chain should be added when needed.
     *
     * TODO (Megatexture): When the megatexture system is implemented, this method
     *   should check `options.atlasHint` and route to MegatextureAllocator instead
     *   of creating a standalone GPUTexture.  The allocator will return a region
     *   descriptor (atlasHandle + uvOffset + uvScale) that callers store alongside
     *   the ResourceHandle so MaterialSystem can pack it into MaterialUniforms.
     */
    async loadImageToTexture(url: string, options?: LoadTextureOptions): Promise<ResourceHandle> {
        const label  = options?.label ?? url.split('/').pop() ?? 'image';
        const format = options?.format      ?? 'rgba8unorm';
        const flipY  = options?.flipY       ?? false;

        // 1. Fetch raw bytes.
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`loadImageToTexture: HTTP ${response.status} fetching "${url}"`);
        }
        const blob = await response.blob();

        // 2. Decode to an ImageBitmap.
        // colorSpaceConversion:'none' keeps the raw bytes untouched — the GPU
        // handles colour-space conversion via the colorSpace field below.
        const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none' });
        const { width, height } = bitmap;

        // 3. Allocate the GPU texture.
        const handle = this.createTexture({
            label,
            size:   { width, height },
            format,
            usage:  GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_DST         |
                    GPUTextureUsage.RENDER_ATTACHMENT,
            // mipLevelCount defaults to 1 — no mip chain yet.
        });

        const gpuTexture = this.getTexture(handle)!;

        // 4. Upload the ImageBitmap.
        this._backend.queue.copyExternalImageToTexture(
            { source: bitmap, flipY },
            { texture: gpuTexture, colorSpace: options?.colorSpace ?? 'srgb' },
            { width, height },
        );

        // Release CPU memory as soon as the GPU copy is enqueued.
        bitmap.close();

        this._log.debug(`Loaded image "${label}" ${width}×${height} format=${format} handle=${handle}`);
        return handle;
    }

    /**
     * Upload an in-memory image (Blob, ArrayBuffer, or Uint8Array) to a GPU texture.
     * Used by the glTF loader where image data comes from embedded buffers, not URLs.
     */
    async loadBlobToTexture(
        data: Blob | ArrayBuffer | Uint8Array,
        options?: LoadTextureOptions,
    ): Promise<ResourceHandle> {
        const label  = options?.label ?? 'blob_texture';
        const format = options?.format ?? 'rgba8unorm';
        const flipY  = options?.flipY  ?? false;

        const blob = data instanceof Blob
            ? data
            : new Blob([data instanceof ArrayBuffer ? data : new Uint8Array(data).buffer as ArrayBuffer]);

        const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none' });
        const { width, height } = bitmap;

        const handle = this.createTexture({
            label,
            size:   { width, height },
            format,
            usage:  GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_DST         |
                    GPUTextureUsage.RENDER_ATTACHMENT,
        });

        const gpuTexture = this.getTexture(handle)!;
        this._backend.queue.copyExternalImageToTexture(
            { source: bitmap, flipY },
            { texture: gpuTexture, colorSpace: options?.colorSpace ?? 'srgb' },
            { width, height },
        );
        bitmap.close();

        this._log.debug(`Loaded blob texture "${label}" ${width}×${height} format=${format} handle=${handle}`);
        return handle;
    }

    /**
     * Load six images (one per cube face) and assemble them into a single
     * cube-mapped GPUTexture.
     *
     * @param basePath  Base URL path without the face suffix, e.g. "/textures/sky/sky_"
     * @param ext       File extension including the dot, e.g. ".png"
     *
     * Face naming convention (appended to basePath):
     *   posx, negx, posy, negy, posz, negz
     *
     * Example: basePath="/textures/sky/sky_" ext=".png"
     *   → fetches sky_posx.png, sky_negx.png, sky_posy.png, …
     *
     * The texture is created with dimension:'2d', depthOrArrayLayers:6 and
     * viewed as 'cube'.  All six faces must have identical dimensions.
     */
    async loadCubemapTexture(
        basePath: string,
        ext: string,
        options?: LoadTextureOptions,
    ): Promise<ResourceHandle> {
        const label  = options?.label ?? 'cubemap';
        const format = options?.format ?? 'rgba8unorm';
        const flipY  = options?.flipY ?? false;

        const faceNames = ['posx', 'negx', 'posy', 'negy', 'posz', 'negz'];
        const urls = faceNames.map(f => `${basePath}${f}${ext}`);

        // Fetch and decode all 6 faces in parallel.
        const bitmaps = await Promise.all(
            urls.map(async (url) => {
                const resp = await fetch(url);
                if (!resp.ok) throw new Error(`loadCubemapTexture: HTTP ${resp.status} fetching "${url}"`);
                const blob = await resp.blob();
                return createImageBitmap(blob, { colorSpaceConversion: 'none' });
            }),
        );

        const { width, height } = bitmaps[0]!;

        // Create the GPU texture (2d array with 6 layers — viewable as 'cube').
        const handle = this.createTexture({
            label,
            size:   { width, height, depthOrArrayLayers: 6 },
            format,
            usage:  GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_DST         |
                    GPUTextureUsage.RENDER_ATTACHMENT,
        });

        const gpuTexture = this.getTexture(handle)!;

        // Upload each face to its layer.
        for (let i = 0; i < 6; i++) {
            this._backend.queue.copyExternalImageToTexture(
                { source: bitmaps[i]!, flipY },
                {
                    texture:  gpuTexture,
                    origin:   { x: 0, y: 0, z: i },
                    colorSpace: options?.colorSpace ?? 'srgb',
                },
                { width, height },
            );
            bitmaps[i]!.close();
        }

        this._log.debug(`Loaded cubemap "${label}" ${width}×${height} format=${format} handle=${handle}`);
        return handle;
    }

    destroyTexture(handle: ResourceHandle): void {
        const tracked = this._textures.get(handle);
        if (!tracked) return;
        tracked.resource.destroy();
        this._textures.delete(handle);
        this._log.debug(`Destroyed texture "${tracked.label}" handle=${handle}`);
    }

    // -------------------------------------------------------------------------
    // Samplers
    // -------------------------------------------------------------------------

    createSampler(desc: SamplerDescriptor): ResourceHandle {
        const label   = desc.label ?? `sampler_${this._nextHandle}`;
        const sampler = this._backend.device.createSampler({
            label,
            addressModeU: desc.addressModeU ?? 'repeat',
            addressModeV: desc.addressModeV ?? 'repeat',
            magFilter:    desc.magFilter    ?? 'linear',
            minFilter:    desc.minFilter    ?? 'linear',
            mipmapFilter: desc.mipmapFilter ?? 'linear',
            maxAnisotropy: desc.maxAnisotropy ?? 1,
            compare:      desc.compare,
        });

        const handle = this._nextHandle++;
        this._samplers.set(handle, { resource: sampler, label });
        this._log.debug(`Created sampler "${label}" handle=${handle}`);
        return handle;
    }

    getSampler(handle: ResourceHandle): GPUSampler | undefined {
        return this._samplers.get(handle)?.resource;
    }

    destroySampler(handle: ResourceHandle): void {
        this._samplers.delete(handle);
    }

    // -------------------------------------------------------------------------
    // Bind Group Layouts (deduplicated by stable JSON hash)
    // -------------------------------------------------------------------------

    /**
     * Returns a cached layout if one with identical entries already exists,
     * otherwise creates and caches a new one.
     */
    getOrCreateBindGroupLayout(entries: GPUBindGroupLayoutEntry[], label?: string): GPUBindGroupLayout {
        const key    = _bindGroupLayoutKey(entries);
        const cached = this._bindGroupLayouts.get(key);
        if (cached) return cached;

        const layout = this._backend.device.createBindGroupLayout({
            label:   label ?? `bgl_${this._bindGroupLayouts.size}`,
            entries,
        });
        this._bindGroupLayouts.set(key, layout);
        this._log.debug(`Created BindGroupLayout "${label ?? key.slice(0, 40)}…"`);
        return layout;
    }

    /**
     * Create a bind group from a layout + resource entries.
     */
    createBindGroup(layout: GPUBindGroupLayout, entries: GPUBindGroupEntry[], label?: string): GPUBindGroup {
        return this._backend.device.createBindGroup({
            label: label ?? 'bind_group',
            layout,
            entries,
        });
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    /**
     * Destroy every GPU resource. Called on engine shutdown.
     */
    destroyAll(): void {
        for (const tracked of this._buffers.values())  tracked.resource.destroy();
        for (const tracked of this._textures.values()) tracked.resource.destroy();
        // Destroy pooled textures too
        for (const pool of this._texturePool.values()) {
            for (const tex of pool) tex.destroy();
        }

        this._buffers.clear();
        this._textures.clear();
        this._samplers.clear();
        this._bindGroupLayouts.clear();
        this._texturePool.clear();

        this._log.info('All GPU resources released');
    }

    // -------------------------------------------------------------------------
    // Stats (useful for debugging / profiling overlays)
    // -------------------------------------------------------------------------

    getStats() {
        let pooledCount = 0;
        for (const pool of this._texturePool.values()) pooledCount += pool.length;
        return {
            buffers:  this._buffers.size,
            textures: this._textures.size,
            samplers: this._samplers.size,
            bindGroupLayouts: this._bindGroupLayouts.size,
            pooledTextures: pooledCount,
        };
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    private _returnToPool(tracked: TrackedTexture): void {
        let pool = this._texturePool.get(tracked.poolKey);
        if (!pool) {
            pool = [];
            this._texturePool.set(tracked.poolKey, pool);
        }
        pool.push(tracked.resource);
    }
}

// -------------------------------------------------------------------------
// Pure helper functions (module-level, no `this` dependency)
// -------------------------------------------------------------------------

/**
 * Stable string key for a texture descriptor — used to match pooled textures.
 * Two descriptors with identical format/size/usage/mip/sample/dimension
 * will produce the same key.
 */
function _texturePoolKey(desc: TextureDescriptor): string {
    // GPUExtent3DStrict can be an array [w, h?, d?] or a dict { width, height?, ... }.
    // Normalise to individual values so the pool key is always correct.
    let w: number, h: number, d: number;
    if (Array.isArray(desc.size)) {
        w = desc.size[0]!;
        h = desc.size[1] ?? 1;
        d = desc.size[2] ?? 1;
    } else {
        const s = desc.size as GPUExtent3DDictStrict;
        w = s.width;
        h = s.height ?? 1;
        d = s.depthOrArrayLayers ?? 1;
    }
    return [
        desc.format,
        w,
        h,
        d,
        desc.usage,
        desc.mipLevelCount  ?? 1,
        desc.sampleCount    ?? 1,
        desc.dimension      ?? '2d',
    ].join('|');
}

/**
 * Stable hash for a bind group layout entry array.
 * Sorts entries by binding index so order doesn't matter.
 */
function _bindGroupLayoutKey(entries: GPUBindGroupLayoutEntry[]): string {
    const sorted = [...entries].sort((a, b) => a.binding - b.binding);
    return JSON.stringify(sorted);
}
