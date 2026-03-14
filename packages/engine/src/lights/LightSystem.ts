// /src/engine/lights/LightSystem.ts

import { mat4 } from 'gl-matrix';
import type { GPUBackend } from '../core/GPUBackend';
import type { ResourceManager, ResourceHandle } from '../core/ResourceManager';
import type { Mat4, Vec3f } from '../scene/SceneGraph';
import type { EngineConfiguration } from '../core/EngineConfiguration';

// -------------------------------------------------------------------------
// Enums & constants
// -------------------------------------------------------------------------

export type LightHandle = number;

export enum LightType {
    Directional = 'DIRECTIONAL',
    Point       = 'POINT',
    Spot        = 'SPOT',
    Area        = 'AREA', // future
}

/**
 * Shadow algorithm to use for a shadow-casting light.
 *
 *   None      — no shadow (default when castShadow=false)
 *   Standard  — single depth map (spot lights)
 *   Cascaded  — Cascaded Shadow Maps / CSM (directional lights)
 *   Cube      — omnidirectional 6-face cube map (point lights)
 */
export enum ShadowType {
    None     = 0,
    Standard = 1,
    Cascaded = 2,
    Cube     = 3,
}

export const MAX_LIGHTS             = 256;
export const MAX_LIGHTS_PER_CLUSTER = 32;
export const SHADOW_ATLAS_SIZE      = 4096;
export const MAX_CSM_CASCADES       = 4;   // up to 4 CSM cascades per directional light
// Maximum shadow cascade/face slots per light (cube = 6 faces > CSM max of 4).
const MAX_SHADOW_CASCADES_PER_LIGHT = 6;
export const CLUSTER_X              = 16;
export const CLUSTER_Y              = 9;
export const CLUSTER_Z              = 24;

// Per-light GPU struct: 5 × vec4f = 80 bytes.
const LIGHT_STRIDE = 80;

// Per-light shadow data buffer (592 bytes per light, 16-byte aligned).
// Layout mirrors the WGSL ShadowData struct in deferred_lighting.wgsl:
//   cascades[6] (CascadeInfo × 6 = 576 bytes) + params vec4f (16 bytes) = 592 bytes
//   (6 slots covers both cube-map faces and CSM, which uses at most 4)
//
//   CascadeInfo (96 bytes):
//     viewProj  : mat4x4f   @ +0   (64 bytes) — light VP for this cascade/face
//     atlasUV   : vec4f     @ +64  (16 bytes) — xy=UV offset, zw=UV scale in atlas
//     params    : vec4f     @ +80  (16 bytes) — x=cascade far eye-depth, y=NDC bias
const SHADOW_DATA_STRIDE = 592;

const TOTAL_CLUSTERS = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;

// -------------------------------------------------------------------------
// Light descriptors
// -------------------------------------------------------------------------

export interface LightDescriptor {
    label?: string;
    type: LightType;
    color: [number, number, number];
    intensity: number;   // luminous power (cd) or illuminance (lux) depending on type
    range?: number;      // point / spot attenuation range
    innerConeAngle?: number; // spot (radians)
    outerConeAngle?: number; // spot (radians)
    castShadow?: boolean;
    shadowBias?: number;
    shadowMapResolution?: number; // per cascade/face in pixels (default 512)
    /**
     * Shadow algorithm for this light.
     * Default: ShadowType.Cascaded for directional, ShadowType.Standard for spot,
     *          ShadowType.Cube for point lights (omnidirectional 6-face shadow map).
     */
    shadowType?: ShadowType;
    /** Number of CSM cascade slices (1–4, directional lights only). Default: 3. */
    numCascades?: number;
    /**
     * PCF filter radius (number of taps in each direction).
     * 1 = 3×3 kernel (9 taps), 2 = 5×5 (25 taps), 3 = 7×7 (49 taps).
     * Higher = softer shadows but more expensive.
     * @default 1
     */
    pcfRadius?: number;
}

/** Per-cascade shadow map state. */
interface CascadeRecord {
    /** Light view-projection matrix for this cascade. */
    viewProj: Mat4;
    /** Sub-region of the shadow atlas allocated to this cascade. */
    atlasRegion: { x: number; y: number; w: number; h: number };
    /** Camera eye-space far depth of this cascade (positive value). */
    splitFar: number;
    /** World-space depth range of the ortho/perspective frustum (far − near). */
    depthRange: number;
    /**
     * Pre-computed NDC bias uploaded to the GPU (casc.params.y in the shader).
     * For CSM: shadowBias * (2R / resolution) / depthRange — shadowBias is a
     *   dimensionless texel multiplier so the bias scales with cascade footprint.
     * For spot/cube: shadowBias / depthRange — shadowBias is a world-space offset.
     */
    ndcBias: number;
    /** Near plane of the shadow projection (used for depth-correct bias). */
    projNear: number;
    /** Far plane of the shadow projection (used for depth-correct bias). */
    projFar: number;
}

/**
 * Public snapshot of a shadow-casting light's cascade data.
 * Returned by getShadowCasterInfos() for the FrameOrchestrator.
 */
export interface ShadowCasterInfo {
    handle:      LightHandle;
    type:        LightType;
    shadowType:  ShadowType;
    shadowBias:  number;
    cascades:    Readonly<CascadeRecord>[];
}

/**
 * Internal light record.
 */
export interface LightRecord {
    handle: LightHandle;
    label: string;
    type: LightType;
    color: [number, number, number];
    intensity: number;
    range: number;
    innerConeAngle: number;
    outerConeAngle: number;
    castShadow: boolean;
    shadowBias: number;
    shadowMapResolution: number;
    shadowType: ShadowType;
    numCascades: number;
    /** PCF filter radius: 1 = 3×3, 2 = 5×5, 3 = 7×7. */
    pcfRadius: number;

    /** World-space position — set each frame via setLightTransform(). */
    worldPosition: Vec3f;
    /** World-space direction — set each frame via setLightTransform(). */
    worldDirection: Vec3f;

    /** Per-cascade shadow data (1 for spot, N for CSM directional). */
    cascades: CascadeRecord[];

    dirty: boolean;
}

/**
 * A cluster bin for clustered light culling.
 */
export interface ClusterBin {
    lightIndices: number[];
}

/**
 * Manages all lights, shadow atlas allocation, per-frame GPU uploads,
 * and clustered deferred light culling data.
 *
 * Clustering scheme: CLUSTER_X × CLUSTER_Y × CLUSTER_Z = 16 × 9 × 24 = 3 456 bins.
 * Z is subdivided exponentially so near clusters are thin and far ones are thick.
 */
export class LightSystem {

    private _backend!: GPUBackend;
    private _resources!: ResourceManager;
    private _nextHandle: LightHandle = 1;
    private _lights: Map<LightHandle, LightRecord> = new Map();

    // --- Configurable limits (from EngineConfiguration) -----------------------
    private _maxLights:              number = MAX_LIGHTS;
    private _maxLightsPerCluster:    number = MAX_LIGHTS_PER_CLUSTER;
    private _shadowAtlasSize:        number = SHADOW_ATLAS_SIZE;
    private _defaultShadowMapRes:    number = 512;
    private _defaultCsmCascades:     number = 3;
    private _defaultShadowBias:      number = 1.5;

    // --- GPU resources --------------------------------------------------------
    /** Structured storage buffer holding per-light data for shaders. */
    private _lightStorageBuffer: ResourceHandle = 0;
    /** Shadow atlas depth texture. */
    private _shadowAtlasTexture: ResourceHandle = 0;
    /** Storage buffer for per-cluster light index lists. */
    private _clusterBuffer: ResourceHandle = 0;

    // --- Clusters (CPU-side, rebuilt each frame) ---
    private _clusters: ClusterBin[] = [];

    // --- Shadow atlas strip allocator -----------------------------------------
    private _atlasX: number = 0;
    private _atlasY: number = 0;
    private _atlasRowHeight: number = 0;

    // --- CPU scratch buffers (allocated once, reused every frame) -------------
    // Float32Array view for light data (header reinterpreted via Uint32Array for count).
    private _lightDataScratch: Float32Array = new Float32Array(0);
    // Uint32Array for cluster index lists.
    private _clusterDataScratch: Uint32Array = new Uint32Array(0);
    // Float32Array for shadow data (SHADOW_DATA_STRIDE bytes per light).
    private _shadowDataBuffer:  ResourceHandle = 0;
    private _shadowDataScratch: Float32Array   = new Float32Array(0);

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    init(backend: GPUBackend, resources: ResourceManager, config?: EngineConfiguration): void {
        this._backend   = backend;
        this._resources = resources;

        // Apply configuration overrides.
        if (config) {
            this._maxLights           = config.maxLights;
            this._maxLightsPerCluster = config.maxLightsPerCluster;
            this._shadowAtlasSize     = config.shadowAtlasSize;
            this._defaultShadowMapRes = config.defaultShadowMapResolution;
            this._defaultCsmCascades  = config.defaultCsmCascades;
            this._defaultShadowBias   = config.defaultShadowBias;
        }

        // Derived sizes from configurable limits.
        const lightBufferSize   = 16 + this._maxLights * LIGHT_STRIDE;
        const shadowBufferSize  = this._maxLights * SHADOW_DATA_STRIDE;
        const clusterEntryInts  = 1 + this._maxLightsPerCluster;
        const totalClusters     = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;
        const clusterBufferSize = totalClusters * clusterEntryInts * 4;

        // 1. Light storage buffer (STORAGE | COPY_DST).
        this._lightStorageBuffer = this._resources.createBuffer({
            label: 'light_storage_buffer',
            size:  lightBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // 2. Cluster buffer (STORAGE | COPY_DST).
        this._clusterBuffer = this._resources.createBuffer({
            label: 'cluster_buffer',
            size:  clusterBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // 3. Shadow atlas texture (depth32float, RENDER_ATTACHMENT | TEXTURE_BINDING).
        this._shadowAtlasTexture = this._resources.createTexture({
            label:  'shadow_atlas',
            size:   { width: this._shadowAtlasSize, height: this._shadowAtlasSize },
            format: 'depth32float',
            usage:  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        // 4. Shadow data storage buffer (STORAGE | COPY_DST).
        this._shadowDataBuffer = this._resources.createBuffer({
            label: 'shadow_data_buffer',
            size:  shadowBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // 5. Pre-size CPU scratch buffers.
        this._lightDataScratch   = new Float32Array(4 + this._maxLights * (LIGHT_STRIDE / 4));
        this._clusterDataScratch = new Uint32Array(totalClusters * clusterEntryInts);
        this._shadowDataScratch  = new Float32Array(shadowBufferSize / 4);

        // 6. Initialise empty CPU cluster grid.
        for (let i = 0; i < totalClusters; i++) {
            this._clusters.push({ lightIndices: [] });
        }
    }

    // -------------------------------------------------------------------------
    // Light CRUD
    // -------------------------------------------------------------------------

    createLight(desc: LightDescriptor): LightHandle {
        const handle = this._nextHandle++;

        // Determine shadow algorithm.
        const castShadow = desc.castShadow ?? false;
        let shadowType = ShadowType.None;
        if (castShadow) {
            if (desc.shadowType !== undefined) {
                shadowType = desc.shadowType;
            } else if (desc.type === LightType.Directional) {
                shadowType = ShadowType.Cascaded;
            } else if (desc.type === LightType.Spot) {
                shadowType = ShadowType.Standard;
            } else if (desc.type === LightType.Point) {
                shadowType = ShadowType.Cube;
            }
        }

        const numCascades =
            shadowType === ShadowType.Cascaded
                ? Math.max(1, Math.min(MAX_CSM_CASCADES, desc.numCascades ?? this._defaultCsmCascades))
                : shadowType === ShadowType.Standard ? 1
                : shadowType === ShadowType.Cube ? 6
                : 0;

        const resolution = desc.shadowMapResolution ?? this._defaultShadowMapRes;

        // Allocate atlas sub-regions — one per cascade / shadow face.
        const cascades: CascadeRecord[] = [];
        for (let i = 0; i < numCascades; i++) {
            const region = this._allocateShadowAtlasRegion(resolution)
                ?? { x: 0, y: 0, w: 0, h: 0 };
            const vp = new Float32Array(16) as Mat4;
            vp[0] = vp[5] = vp[10] = vp[15] = 1; // identity
            cascades.push({ viewProj: vp, atlasRegion: region, splitFar: 0, depthRange: 1.0, ndcBias: 0, projNear: 0, projFar: 1 });
        }

        const record: LightRecord = {
            handle,
            label:               desc.label              ?? `light_${handle}`,
            type:                desc.type,
            color:               [...desc.color]          as [number, number, number],
            intensity:           desc.intensity,
            range:               desc.range              ?? 10.0,
            innerConeAngle:      desc.innerConeAngle      ?? 0.0,
            outerConeAngle:      desc.outerConeAngle      ?? Math.PI / 4,
            castShadow,
            shadowBias:          desc.shadowBias
                                 ?? (desc.type === LightType.Directional
                                     ? this._defaultShadowBias   // texel multiplier (1.0–2.0)
                                     : 0.005),                   // world-space offset for spot/cube
            shadowMapResolution: resolution,
            shadowType,
            numCascades,
            pcfRadius:           Math.max(0, Math.min(3, desc.pcfRadius ?? 1)),
            worldPosition:       new Float32Array([0, 0, 0]),
            worldDirection:      new Float32Array([0, -1, 0]),
            cascades,
            dirty:               true,
        };

        this._lights.set(handle, record);
        return handle;
    }

    getLight(handle: LightHandle): LightRecord | undefined {
        return this._lights.get(handle);
    }

    /** Iterate all registered lights. */
    getLights(): IterableIterator<LightRecord> {
        return this._lights.values();
    }

    /**
     * Set the world-space position and direction for a light's scene node.
     * Must be called every frame after SceneGraph.updateWorldMatrices().
     */
    setLightTransform(handle: LightHandle, worldPosition: Vec3f, worldDirection: Vec3f): void {
        const light = this._lights.get(handle);
        if (!light) return;
        light.worldPosition[0]  = worldPosition[0]!;
        light.worldPosition[1]  = worldPosition[1]!;
        light.worldPosition[2]  = worldPosition[2]!;
        light.worldDirection[0] = worldDirection[0]!;
        light.worldDirection[1] = worldDirection[1]!;
        light.worldDirection[2] = worldDirection[2]!;
        light.dirty = true;
    }

    updateLight(handle: LightHandle, changes: Partial<LightDescriptor>): void {
        const light = this._lights.get(handle);
        if (!light) return;
        if (changes.color            !== undefined) light.color            = [...changes.color] as [number, number, number];
        if (changes.intensity        !== undefined) light.intensity        = changes.intensity;
        if (changes.range            !== undefined) light.range            = changes.range;
        if (changes.innerConeAngle   !== undefined) light.innerConeAngle   = changes.innerConeAngle;
        if (changes.outerConeAngle   !== undefined) light.outerConeAngle   = changes.outerConeAngle;
        if (changes.shadowBias       !== undefined) light.shadowBias       = changes.shadowBias;
        if (changes.pcfRadius        !== undefined) light.pcfRadius        = Math.max(0, Math.min(3, changes.pcfRadius));
        light.dirty = true;
    }

    destroyLight(handle: LightHandle): void {
        // Shadow atlas regions are not reclaimed at runtime (simple allocator).
        this._lights.delete(handle);
    }

    // -------------------------------------------------------------------------
    // Shadow maps
    // -------------------------------------------------------------------------

    /**
     * Compute shadow view-projection matrix(es) for a light.
     * Must be called after setLightTransform() each frame.
     *
     * For CSM (directional lights) cameraParams must be supplied so the
     * cascade frustums can be derived from the active camera.
     */
    computeShadowMatrices(
        handle: LightHandle,
        cameraParams?: {
            viewMatrix: Mat4;
            projMatrix: Mat4;
            near:       number;
            far:        number;
        },
    ): void {
        const light = this._lights.get(handle);
        if (!light || !light.castShadow) return;

        if (light.shadowType === ShadowType.Cascaded && cameraParams) {
            this._computeCSMMatrices(light, cameraParams);
        } else if (light.shadowType === ShadowType.Standard) {
            this._computeSpotShadowMatrix(light);
        } else if (light.shadowType === ShadowType.Cube) {
            this._computePointShadowMatrices(light);
        }
        light.dirty = true;
    }

    getShadowAtlasTexture(): ResourceHandle {
        return this._shadowAtlasTexture;
    }

    /** Return list of shadow-casting lights that need a shadow pass this frame. */
    getShadowCasters(): LightHandle[] {
        const casters: LightHandle[] = [];
        for (const [, light] of this._lights) {
            if (light.castShadow) casters.push(light.handle);
        }
        return casters;
    }

    /** Public snapshot of all shadow-casting lights (for FrameOrchestrator). */
    getShadowCasterInfos(): ShadowCasterInfo[] {
        const result: ShadowCasterInfo[] = [];
        for (const [, light] of this._lights) {
            if (light.castShadow && light.shadowType !== ShadowType.None) {
                result.push({
                    handle:     light.handle,
                    type:       light.type,
                    shadowType: light.shadowType,
                    shadowBias: light.shadowBias,
                    cascades:   light.cascades,
                });
            }
        }
        return result;
    }

    /**
     * Pack all shadow cascade data into the GPU shadow data storage buffer.
     *
     * Per-light layout (SHADOW_DATA_STRIDE = 592 bytes = 148 floats):
     *   cascades[6] × 96 bytes (24 floats each):
     *     viewProj  : mat4x4f  @ +0   (floats 0–15)
     *     atlasUV   : vec4f    @ +64  (floats 16–19) — xy=UV offset, zw=UV scale
     *     params    : vec4f    @ +80  (floats 20–23) — x=cascade far depth, y=NDC bias
     *   params      : vec4f    @ +576 (floats 144–147) — x=shadowType, y=numCascades, z=atlasTexelSize, w=pcfRadius
     */
    uploadShadowData(): void {
        if (this._shadowDataBuffer === 0) return;

        const scratch = this._shadowDataScratch;
        let li = 0;
        for (const [, light] of this._lights) {
            if (li >= this._maxLights) break;
            const base = li * (SHADOW_DATA_STRIDE / 4); // float index

            for (let ci = 0; ci < MAX_SHADOW_CASCADES_PER_LIGHT; ci++) {
                const cb   = base + ci * 24; // 24 floats per cascade
                const casc = light.cascades[ci];
                if (casc) {
                    for (let j = 0; j < 16; j++) scratch[cb + j] = casc.viewProj[j] ?? 0;
                    scratch[cb + 16] = casc.atlasRegion.x / this._shadowAtlasSize;
                    scratch[cb + 17] = casc.atlasRegion.y / this._shadowAtlasSize;
                    scratch[cb + 18] = casc.atlasRegion.w / this._shadowAtlasSize;
                    scratch[cb + 19] = casc.atlasRegion.h / this._shadowAtlasSize;
                    scratch[cb + 20] = casc.splitFar;
                    scratch[cb + 21] = casc.ndcBias;
                    scratch[cb + 22] = casc.projNear;
                    scratch[cb + 23] = casc.projFar;
                } else {
                    for (let j = 0; j < 24; j++) scratch[cb + j] = 0;
                }
            }

            // Per-light params at float offset 144 (byte offset 576).
            scratch[base + 144] = light.shadowType;
            scratch[base + 145] = light.numCascades;
            scratch[base + 146] = 1.0 / this._shadowAtlasSize; // atlas texel size
            scratch[base + 147] = light.pcfRadius;
            li++;
        }

        const uploadBytes = Math.min(this._lights.size, this._maxLights) * SHADOW_DATA_STRIDE;
        const gpuBuffer   = this._resources.getBuffer(this._shadowDataBuffer);
        if (gpuBuffer && uploadBytes > 0) {
            this._backend.queue.writeBuffer(gpuBuffer, 0, scratch.buffer, 0, uploadBytes);
        }
    }

    getShadowDataBuffer(): ResourceHandle {
        return this._shadowDataBuffer;
    }

    // -------------------------------------------------------------------------
    // Clustered light culling
    // -------------------------------------------------------------------------

    /**
     * Assign lights to cluster bins based on the camera frustum subdivision.
     * Call once per frame after world-matrix propagation, before the lighting pass.
     *
     * Exponential Z subdivision: depthAtSlice_k = near × (far/near)^(k/CLUSTER_Z).
     * View-space AABB per cluster is computed by un-projecting NDC corners at
     * the near and far Z of each slice.
     *
     * @param viewMatrix       Active camera view matrix (column-major)
     * @param projectionMatrix Active camera projection matrix (column-major, perspectiveZO)
     */
    buildClusters(viewMatrix: Mat4, projectionMatrix: Mat4): void {
        // Extract near/far from a gl-matrix perspectiveZO column-major matrix.
        //   proj[10] = far  / (near - far)
        //   proj[14] = far * near / (near - far)
        //   → near = proj[14] / proj[10]
        //   → far  = proj[14] / (proj[10] + 1)
        const m10  = projectionMatrix[10]!;
        const m14  = projectionMatrix[14]!;
        if (m10 === 0) return; // not a valid perspective matrix
        const near = m14 / m10;
        const far  = m14 / (m10 + 1.0);
        if (near <= 0 || far <= near) return;

        // Projection-space scale factors used to unproject NDC → view space.
        const projX = projectionMatrix[0]!; // f / aspect
        const projY = projectionMatrix[5]!; // f

        // Reset cluster bins.
        for (let i = 0; i < TOTAL_CLUSTERS; i++) {
            this._clusters[i]!.lightIndices.length = 0;
        }

        // Collect all active lights, transforming their world position to view space.
        type ActiveLight = {
            lightIndex: number;
            viewX: number; viewY: number; viewZ: number; // view-space position
            range: number;
            type: LightType;
        };
        const activeLights: ActiveLight[] = [];
        let lightIndex = 0;
        for (const [, light] of this._lights) {
            const wp = light.worldPosition;
            // Column-major mat4 × point: result = M × [x,y,z,1]
            const vx = viewMatrix[0]! * wp[0]! + viewMatrix[4]! * wp[1]! + viewMatrix[8]!  * wp[2]! + viewMatrix[12]!;
            const vy = viewMatrix[1]! * wp[0]! + viewMatrix[5]! * wp[1]! + viewMatrix[9]!  * wp[2]! + viewMatrix[13]!;
            const vz = viewMatrix[2]! * wp[0]! + viewMatrix[6]! * wp[1]! + viewMatrix[10]! * wp[2]! + viewMatrix[14]!;
            activeLights.push({ lightIndex: lightIndex++, viewX: vx, viewY: vy, viewZ: vz, range: light.range, type: light.type });
        }

        if (activeLights.length === 0) {
            this._uploadClusterBuffer();
            return;
        }

        const logRatio = Math.log(far / near);

        // Iterate all clusters and assign overlapping lights.
        for (let zi = 0; zi < CLUSTER_Z; zi++) {
            // Exponential Z slice.
            const zNear = near * Math.exp(logRatio * zi       / CLUSTER_Z);
            const zFar  = near * Math.exp(logRatio * (zi + 1) / CLUSTER_Z);

            for (let yi = 0; yi < CLUSTER_Y; yi++) {
                // NDC y: top = +1, bottom = −1 (y flipped relative to screen y).
                const ndcYMin = 1.0 - 2.0 * (yi + 1) / CLUSTER_Y;
                const ndcYMax = 1.0 - 2.0 * yi       / CLUSTER_Y;

                for (let xi = 0; xi < CLUSTER_X; xi++) {
                    const ndcXMin = 2.0 * xi       / CLUSTER_X - 1.0;
                    const ndcXMax = 2.0 * (xi + 1) / CLUSTER_X - 1.0;

                    // Unproject NDC corners at both slice depths → view-space AABB.
                    // In RH view space z is negative; depth = −z.
                    // viewX = ndcX × depth / projX,  viewY = ndcY × depth / projY
                    const xMinN = ndcXMin * zNear / projX; const xMaxN = ndcXMax * zNear / projX;
                    const yMinN = ndcYMin * zNear / projY; const yMaxN = ndcYMax * zNear / projY;
                    const xMinF = ndcXMin * zFar  / projX; const xMaxF = ndcXMax * zFar  / projX;
                    const yMinF = ndcYMin * zFar  / projY; const yMaxF = ndcYMax * zFar  / projY;

                    const aabbMinX = Math.min(xMinN, xMaxN, xMinF, xMaxF);
                    const aabbMaxX = Math.max(xMinN, xMaxN, xMinF, xMaxF);
                    const aabbMinY = Math.min(yMinN, yMaxN, yMinF, yMaxF);
                    const aabbMaxY = Math.max(yMinN, yMaxN, yMinF, yMaxF);
                    // View-space z is negative; zNear/zFar are positive depths.
                    const aabbMinZ = -zFar;
                    const aabbMaxZ = -zNear;

                    const clusterIdx = zi * CLUSTER_X * CLUSTER_Y + yi * CLUSTER_X + xi;
                    const bin        = this._clusters[clusterIdx]!;

                    for (const al of activeLights) {
                        if (bin.lightIndices.length >= MAX_LIGHTS_PER_CLUSTER) break;

                        if (al.type === LightType.Directional) {
                            // Directional lights illuminate every cluster.
                            bin.lightIndices.push(al.lightIndex);
                        } else if (_sphereIntersectsAABB(
                            al.viewX, al.viewY, al.viewZ, al.range,
                            aabbMinX, aabbMinY, aabbMinZ,
                            aabbMaxX, aabbMaxY, aabbMaxZ,
                        )) {
                            bin.lightIndices.push(al.lightIndex);
                        }
                    }
                }
            }
        }

        this._uploadClusterBuffer();
    }

    getClusterBuffer(): ResourceHandle {
        return this._clusterBuffer;
    }

    // -------------------------------------------------------------------------
    // Per-frame GPU upload
    // -------------------------------------------------------------------------

    /**
     * Pack all active LightRecords into the GPU storage buffer.
     *
     * Per-light layout (80 bytes = 5 × vec4f):
     *   offset  0  position.xyz,   range.w
     *   offset 16  color.rgb,      intensity.w
     *   offset 32  direction.xyz,  type.w   (0=directional, 1=point, 2=spot)
     *   offset 48  innerConeCos.x, outerConeCos.y, shadowBias.z, castShadow.w
     *   offset 64  shadowAtlasUVOffset.xy, shadowAtlasUVScale.zw
     *
     * Buffer header (16 bytes): u32 lightCount, u32 pad×3
     */
    uploadLightData(): void {
        if (this._lightStorageBuffer === 0) return;

        const scratch = this._lightDataScratch;
        const lights  = [...this._lights.values()];
        const count   = Math.min(lights.length, this._maxLights);

        // Write header count as a u32 using a shared-buffer DataView.
        new Uint32Array(scratch.buffer, 0, 1)[0] = count;

        for (let i = 0; i < count; i++) {
            const light = lights[i]!;
            // Float32 index: header occupies 4 floats (16 bytes), then 20 floats per light.
            const b = 4 + i * (LIGHT_STRIDE / 4);

            // vec4: position.xyz + range.w
            scratch[b + 0] = light.worldPosition[0]!;
            scratch[b + 1] = light.worldPosition[1]!;
            scratch[b + 2] = light.worldPosition[2]!;
            scratch[b + 3] = light.range;

            // vec4: color.rgb + intensity.w
            scratch[b + 4] = light.color[0]!;
            scratch[b + 5] = light.color[1]!;
            scratch[b + 6] = light.color[2]!;
            scratch[b + 7] = light.intensity;

            // vec4: direction.xyz (pre-normalized) + type.w
            const dx = light.worldDirection[0]!;
            const dy = light.worldDirection[1]!;
            const dz = light.worldDirection[2]!;
            const dLen = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
            scratch[b + 8]  = dx / dLen;
            scratch[b + 9]  = dy / dLen;
            scratch[b + 10] = dz / dLen;
            scratch[b + 11] = _lightTypeToInt(light.type);

            // vec4: innerConeCos, outerConeCos, shadowBias, castShadow
            scratch[b + 12] = Math.cos(light.innerConeAngle);
            scratch[b + 13] = Math.cos(light.outerConeAngle);
            scratch[b + 14] = light.shadowBias;
            scratch[b + 15] = light.castShadow ? 1.0 : 0.0;

            // vec4: shadow atlas UV offset + scale (first cascade, for backward compat)
            const region = light.cascades[0]?.atlasRegion;
            if (region && region.w > 0) {
                scratch[b + 16] = region.x / this._shadowAtlasSize;
                scratch[b + 17] = region.y / this._shadowAtlasSize;
                scratch[b + 18] = region.w / this._shadowAtlasSize;
                scratch[b + 19] = region.h / this._shadowAtlasSize;
            } else {
                scratch[b + 16] = 0; scratch[b + 17] = 0;
                scratch[b + 18] = 0; scratch[b + 19] = 0;
            }

            light.dirty = false;
        }

        // Upload only the populated portion.
        const uploadBytes = 16 + count * LIGHT_STRIDE;
        const gpuBuffer   = this._resources.getBuffer(this._lightStorageBuffer);
        if (gpuBuffer) {
            this._backend.queue.writeBuffer(gpuBuffer, 0, scratch.buffer, 0, uploadBytes);
        }
    }

    getLightStorageBuffer(): ResourceHandle {
        return this._lightStorageBuffer;
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    destroy(): void {
        if (this._lightStorageBuffer !== 0) {
            this._resources.destroyBuffer(this._lightStorageBuffer);
            this._lightStorageBuffer = 0;
        }
        if (this._clusterBuffer !== 0) {
            this._resources.destroyBuffer(this._clusterBuffer);
            this._clusterBuffer = 0;
        }
        if (this._shadowAtlasTexture !== 0) {
            this._resources.destroyTexture(this._shadowAtlasTexture);
            this._shadowAtlasTexture = 0;
        }
        if (this._shadowDataBuffer !== 0) {
            this._resources.destroyBuffer(this._shadowDataBuffer);
            this._shadowDataBuffer = 0;
        }
        this._lights.clear();
        this._clusters.length = 0;
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * Build CSM view-projection matrices for a directional light.
     * Uses a practical log-linear cascade split (λ=0.85) and texel snapping
     * to prevent shadow map swimming as the camera moves.
     */
    private _computeCSMMatrices(
        light: LightRecord,
        cam: { viewMatrix: Mat4; projMatrix: Mat4; near: number; far: number },
    ): void {
        const N  = light.numCascades;
        const λ  = 0.85;
        const nearPad = 50.0; // extend behind camera so rear casters are captured

        // 1. Build orthonormal light-space basis.
        const [dx, dy, dz] = _normalise3(
            light.worldDirection[0]!, light.worldDirection[1]!, light.worldDirection[2]!,
        );
        const [ux, uy, uz]: [number, number, number] =
            Math.abs(dy) > 0.99 ? [1, 0, 0] : [0, 1, 0];
        const [rx, ry, rz] = _normalise3(
            uy * dz - uz * dy, uz * dx - ux * dz, ux * dy - uy * dx,
        );
        // lightUp = cross(lightDir, right) — already unit length.
        const lux = dy * rz - dz * ry;
        const luy = dz * rx - dx * rz;
        const luz = dx * ry - dy * rx;

        // 2. Invert camera VP to unproject NDC corners → world space.
        const camVP = new Float32Array(16) as Mat4;
        const invVP = new Float32Array(16) as Mat4;
        mat4.multiply(camVP, cam.projMatrix, cam.viewMatrix);
        if (!mat4.invert(invVP, camVP)) return;

        // Full-frustum NDC corners (WebGPU: z ∈ [0,1]).
        const wsNear: [number, number, number][] = [
            _unprojectNDC(-1, -1, 0, invVP), _unprojectNDC(1, -1, 0, invVP),
            _unprojectNDC(-1,  1, 0, invVP), _unprojectNDC(1,  1, 0, invVP),
        ];
        const wsFar: [number, number, number][] = [
            _unprojectNDC(-1, -1, 1, invVP), _unprojectNDC(1, -1, 1, invVP),
            _unprojectNDC(-1,  1, 1, invVP), _unprojectNDC(1,  1, 1, invVP),
        ];

        // 3. Per-cascade ortho VP.
        const { near, far } = cam;
        let prevSplit = near;
        for (let ci = 0; ci < N; ci++) {
            const t        = (ci + 1) / N;
            const logSplit = near * Math.pow(far / near, t);
            const linSplit = near + (far - near) * t;
            const splitFar = λ * logSplit + (1 - λ) * linSplit;

            // Lerp full-frustum corners to isolate this cascade slice.
            const tN = (prevSplit - near) / (far - near);
            const tF = (splitFar  - near) / (far - near);
            let cx = 0, cy = 0, cz = 0;
            const corners: [number, number, number][] = [];
            for (let k = 0; k < 4; k++) {
                const wn = wsNear[k]!, wf = wsFar[k]!;
                const pN: [number, number, number] = [
                    wn[0] + (wf[0] - wn[0]) * tN,
                    wn[1] + (wf[1] - wn[1]) * tN,
                    wn[2] + (wf[2] - wn[2]) * tN,
                ];
                const pF: [number, number, number] = [
                    wn[0] + (wf[0] - wn[0]) * tF,
                    wn[1] + (wf[1] - wn[1]) * tF,
                    wn[2] + (wf[2] - wn[2]) * tF,
                ];
                corners.push(pN, pF);
                cx += pN[0] + pF[0];
                cy += pN[1] + pF[1];
                cz += pN[2] + pF[2];
            }
            cx /= 8; cy /= 8; cz /= 8;

            // Bounding sphere radius in world space.
            let R = 0;
            for (const [px, py, pz] of corners) {
                const d = Math.sqrt((px-cx)*(px-cx) + (py-cy)*(py-cy) + (pz-cz)*(pz-cz));
                R = Math.max(R, d);
            }

            // Texel snapping: quantize center in light-space XY to prevent swimming.
            const texelSize   = (2 * R) / light.shadowMapResolution;
            const origLsCx    = rx * cx + ry * cy + rz * cz;
            const origLsCy    = lux * cx + luy * cy + luz * cz;
            const snappedLsCx = Math.floor(origLsCx / texelSize) * texelSize;
            const snappedLsCy = Math.floor(origLsCy / texelSize) * texelSize;
            const dLsCx       = snappedLsCx - origLsCx;
            const dLsCy       = snappedLsCy - origLsCy;
            const scx = cx + dLsCx * rx + dLsCy * lux;
            const scy = cy + dLsCx * ry + dLsCy * luy;
            const scz = cz + dLsCx * rz + dLsCy * luz;

            // Light view: lookAt from snapped center pulled back along lightDir.
            const eye: [number, number, number] = [
                scx - dx * (R + nearPad),
                scy - dy * (R + nearPad),
                scz - dz * (R + nearPad),
            ];
            const lightView = new Float32Array(16) as Mat4;
            mat4.lookAt(lightView, eye, [scx, scy, scz], [lux, luy, luz]);

            // Orthographic projection (WebGPU z ∈ [0,1]).
            const lightProj = new Float32Array(16) as Mat4;
            mat4.orthoZO(lightProj, -R, R, -R, R, 0.0, 2 * R + nearPad);

            const cascade = light.cascades[ci];
            if (cascade) {
                mat4.multiply(cascade.viewProj, lightProj, lightView);
                cascade.splitFar   = splitFar;
                cascade.depthRange = 2 * R + nearPad;
                // CSM: shadowBias is a texel multiplier so bias scales with cascade footprint.
                // ndcBias = shadowBias * (texelWorldSize / depthRange)
                //         = shadowBias * (2R / resolution) / (2R + nearPad)
                const texelWorld = (2 * R) / light.shadowMapResolution;
                cascade.ndcBias  = light.shadowBias * texelWorld / cascade.depthRange;
                cascade.projNear = 0.0;
                cascade.projFar  = 2 * R + nearPad;
            }
            prevSplit = splitFar;
        }
    }

    /**
     * Build a single perspective shadow VP for a spot light.
     */
    private _computeSpotShadowMatrix(light: LightRecord): void {
        const cascade = light.cascades[0];
        if (!cascade) return;

        const pos = light.worldPosition;
        const [dx, dy, dz] = _normalise3(
            light.worldDirection[0]!, light.worldDirection[1]!, light.worldDirection[2]!,
        );
        const [ux, uy, uz]: [number, number, number] =
            Math.abs(dy) > 0.99 ? [1, 0, 0] : [0, 1, 0];

        const eye: [number, number, number]    = [pos[0]!, pos[1]!, pos[2]!];
        const center: [number, number, number] = [pos[0]! + dx, pos[1]! + dy, pos[2]! + dz];

        const lightView = new Float32Array(16) as Mat4;
        mat4.lookAt(lightView, eye, center, [ux, uy, uz]);

        // FOV = 2 × outerConeAngle (outerConeAngle is the half-angle).
        const lightProj = new Float32Array(16) as Mat4;
        mat4.perspectiveZO(lightProj, light.outerConeAngle * 2, 1.0, 0.1, light.range);

        mat4.multiply(cascade.viewProj, lightProj, lightView);
        cascade.splitFar   = light.range;
        cascade.depthRange = light.range - 0.1; // perspective far − near
        cascade.ndcBias    = light.shadowBias / cascade.depthRange;
        cascade.projNear   = 0.1;
        cascade.projFar    = light.range;
    }

    /**
     * Build 6 perspective shadow VPs for an omnidirectional point light.
     * Each VP covers one face of a virtual cube centered at the light position.
     * FOV = 90°, aspect = 1:1, near = 0.05, far = light.range.
     *
     * Face order (matches GPU face-selection logic in evalShadow):
     *   0: +X   1: −X   2: +Y   3: −Y   4: +Z   5: −Z
     */
    private _computePointShadowMatrices(light: LightRecord): void {
        const near = 0.05;
        const far  = Math.max(light.range, near + 0.1);

        const lightProj = new Float32Array(16) as Mat4;
        mat4.perspectiveZO(lightProj, Math.PI / 2, 1.0, near, far);

        // Each face: [forward direction, up vector]
        const FACES: Array<[[number, number, number], [number, number, number]]> = [
            [[ 1,  0,  0], [0, -1,  0]],  // +X
            [[-1,  0,  0], [0, -1,  0]],  // −X
            [[ 0,  1,  0], [0,  0,  1]],  // +Y
            [[ 0, -1,  0], [0,  0, -1]],  // −Y
            [[ 0,  0,  1], [0, -1,  0]],  // +Z
            [[ 0,  0, -1], [0, -1,  0]],  // −Z
        ];

        const pos = light.worldPosition;
        const eye: [number, number, number] = [pos[0]!, pos[1]!, pos[2]!];

        for (let fi = 0; fi < 6; fi++) {
            const cascade = light.cascades[fi];
            if (!cascade) continue;

            const [dir, up] = FACES[fi]!;
            const center: [number, number, number] = [
                eye[0] + dir[0]!,
                eye[1] + dir[1]!,
                eye[2] + dir[2]!,
            ];

            const lightView = new Float32Array(16) as Mat4;
            mat4.lookAt(lightView, eye, center, up);
            mat4.multiply(cascade.viewProj, lightProj, lightView);
            cascade.splitFar   = far;
            cascade.depthRange = far - near;
            cascade.ndcBias    = light.shadowBias / cascade.depthRange;
            cascade.projNear   = near;
            cascade.projFar    = far;
        }
    }

    /**
     * Allocate a square shadow-map region in the atlas (strip allocator).
     * Rows are packed left-to-right; when a row is full we advance to the next.
     * Returns undefined if the atlas is exhausted.
     */
    private _allocateShadowAtlasRegion(size: number): { x: number; y: number; w: number; h: number } | undefined {
        if (this._atlasX + size > this._shadowAtlasSize) {
            this._atlasX = 0;
            this._atlasY += this._atlasRowHeight;
            this._atlasRowHeight = 0;
        }
        if (this._atlasY + size > this._shadowAtlasSize) return undefined; // atlas full

        const region = { x: this._atlasX, y: this._atlasY, w: size, h: size };
        this._atlasX        += size;
        this._atlasRowHeight = Math.max(this._atlasRowHeight, size);
        return region;
    }

    /**
     * Pack CPU _clusters[] into _clusterDataScratch and upload to the cluster buffer.
     *
     * Per-cluster layout (u32 values):
     *   [0]        lightCount (≤ MAX_LIGHTS_PER_CLUSTER)
     *   [1 .. 32]  light indices (into the per-frame light array)
     */
    private _uploadClusterBuffer(): void {
        if (this._clusterBuffer === 0) return;

        const buf = this._clusterDataScratch;
        const entryInts = 1 + this._maxLightsPerCluster;
        for (let i = 0; i < TOTAL_CLUSTERS; i++) {
            const bin  = this._clusters[i]!;
            const base = i * entryInts;
            const cnt  = Math.min(bin.lightIndices.length, this._maxLightsPerCluster);
            buf[base]  = cnt;
            for (let j = 0; j < cnt; j++) {
                buf[base + 1 + j] = bin.lightIndices[j]!;
            }
        }

        const gpuBuffer = this._resources.getBuffer(this._clusterBuffer);
        if (gpuBuffer) {
            const clusterBufferSize = TOTAL_CLUSTERS * entryInts * 4;
            this._backend.queue.writeBuffer(gpuBuffer, 0, buf.buffer, 0, clusterBufferSize);
        }
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

/**
 * Returns true if a sphere overlaps an axis-aligned bounding box.
 * Uses the closest-point-on-AABB algorithm:
 *   d² = Σ max(min - c, 0, c - max)²  ≤  r²
 */
function _sphereIntersectsAABB(
    cx: number, cy: number, cz: number, radius: number,
    minX: number, minY: number, minZ: number,
    maxX: number, maxY: number, maxZ: number,
): boolean {
    const dx = Math.max(minX - cx, 0, cx - maxX);
    const dy = Math.max(minY - cy, 0, cy - maxY);
    const dz = Math.max(minZ - cz, 0, cz - maxZ);
    return dx * dx + dy * dy + dz * dz <= radius * radius;
}

/** Normalise a 3-component vector; returns [0,0,1] for zero-length input. */
function _normalise3(x: number, y: number, z: number): [number, number, number] {
    const len = Math.sqrt(x * x + y * y + z * z);
    return len > 1e-10 ? [x / len, y / len, z / len] : [0, 0, 1];
}

/**
 * Unproject an NDC point (WebGPU convention: z ∈ [0,1]) to world space.
 * Performs a homogeneous divide after multiplying by invVP.
 */
function _unprojectNDC(
    ndcX: number, ndcY: number, ndcZ: number,
    invVP: Float32Array,
): [number, number, number] {
    const m  = invVP;
    const x  = m[0]!*ndcX + m[4]!*ndcY + m[8]!*ndcZ  + m[12]!;
    const y  = m[1]!*ndcX + m[5]!*ndcY + m[9]!*ndcZ  + m[13]!;
    const z  = m[2]!*ndcX + m[6]!*ndcY + m[10]!*ndcZ + m[14]!;
    const w  = m[3]!*ndcX + m[7]!*ndcY + m[11]!*ndcZ + m[15]!;
    const rw = w !== 0 ? 1 / w : 1;
    return [x * rw, y * rw, z * rw];
}

function _lightTypeToInt(type: LightType): number {
    switch (type) {
        case LightType.Directional: return 0;
        case LightType.Point:       return 1;
        case LightType.Spot:        return 2;
        default:                    return 1;
    }
}
