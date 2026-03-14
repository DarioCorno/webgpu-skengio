// /src/engine/geometry/MeshSystem.ts

import { Logger } from '../core/Logger';
import type { GPUBackend } from '../core/GPUBackend';
import type { ResourceManager, ResourceHandle } from '../core/ResourceManager';

// -------------------------------------------------------------------------
// Math helpers (minimal, expand or replace with a math lib)
// -------------------------------------------------------------------------

export interface Vec3 {
    x: number;
    y: number;
    z: number;
}

export interface AABB {
    min: Vec3;
    max: Vec3;
}

export interface BoundingSphere {
    center: Vec3;
    radius: number;
}

// -------------------------------------------------------------------------
// Vertex layout descriptors
// -------------------------------------------------------------------------

/**
 * Predefined vertex attribute semantics the engine understands.
 */
export enum VertexSemantic {
    Position  = 'POSITION',
    Normal    = 'NORMAL',
    Tangent   = 'TANGENT',
    UV0       = 'TEXCOORD_0',
    UV1       = 'TEXCOORD_1',
    Color0    = 'COLOR_0',
    Joints0   = 'JOINTS_0',
    Weights0  = 'WEIGHTS_0',
}

export interface VertexAttributeDesc {
    semantic: VertexSemantic;
    format: GPUVertexFormat;
    offset: number;
}

export interface VertexLayoutDesc {
    arrayStride: number;
    stepMode?: GPUVertexStepMode;
    attributes: VertexAttributeDesc[];
}

// -------------------------------------------------------------------------
// Sub-mesh / LOD
// -------------------------------------------------------------------------

export interface SubMesh {
    /** First index in the index buffer (indexed draws). */
    indexOffset: number;
    /** Number of indices to draw. 0 for non-indexed draws. */
    indexCount: number;
    /** baseVertex offset for indexed draws; firstVertex for non-indexed. */
    vertexOffset: number;
    /** Vertex count — needed for non-indexed draws. */
    vertexCount: number;
    /** Index into the parent mesh's material slot array. */
    materialIndex: number;
}

export interface LODLevel {
    /** Screen-space threshold below which this LOD is selected (0–1). */
    screenCoverage: number;
    subMeshes: SubMesh[];
}

// -------------------------------------------------------------------------
// Mesh descriptor / handle
// -------------------------------------------------------------------------

export type MeshHandle = number;

export interface MeshDescriptor {
    label?: string;
    vertexLayouts: VertexLayoutDesc[];
    /** Raw vertex data per layout (interleaved or split, one ArrayBuffer per layout). */
    vertexData: ArrayBuffer[];
    indexData?: Uint16Array | Uint32Array;
    lodLevels?: LODLevel[];
    /** If omitted, computed from position data. */
    aabb?: AABB;
}

// -------------------------------------------------------------------------
// Public draw-data accessor
// -------------------------------------------------------------------------

/**
 * Everything a render-pass encoder needs to bind and draw a mesh LOD.
 * Returned by MeshSystem.getDrawData().
 */
export interface MeshDrawData {
    vertexBuffers: GPUBuffer[];
    vertexLayouts: VertexLayoutDesc[];
    indexBuffer:   GPUBuffer | null;
    indexFormat:   GPUIndexFormat;
    lodLevels:     LODLevel[];
    aabb:          AABB;
    boundingSphere: BoundingSphere;
}

// -------------------------------------------------------------------------
// Internal MeshRecord
// -------------------------------------------------------------------------

interface MeshRecord {
    handle: MeshHandle;
    label: string;
    vertexBuffers: ResourceHandle[];
    indexBuffer?: ResourceHandle;
    indexFormat: GPUIndexFormat;
    vertexLayouts: VertexLayoutDesc[];
    lodLevels: LODLevel[];
    aabb: AABB;
    boundingSphere: BoundingSphere;
}

// -------------------------------------------------------------------------
// MeshSystem
// -------------------------------------------------------------------------

/**
 * Manages geometry data: vertex/index buffers, bounding volumes, LODs.
 *
 * Workflow:
 *   const handle = meshSystem.createMesh(descriptor);   // uploads to GPU
 *   const draw   = meshSystem.getDrawData(handle);       // resolve for render
 *   meshSystem.destroyMesh(handle);                      // free GPU buffers
 *
 * Future RT: build BLAS per mesh and expose to the scene graph for TLAS.
 */
export class MeshSystem {

    private _backend!: GPUBackend;
    private _resources!: ResourceManager;
    private _nextHandle: MeshHandle = 1;
    private _meshes: Map<MeshHandle, MeshRecord> = new Map();
    private readonly _log = new Logger('MeshSystem');

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    init(backend: GPUBackend, resources: ResourceManager): void {
        this._backend = backend;
        this._resources = resources;
    }

    // -------------------------------------------------------------------------
    // Mesh CRUD
    // -------------------------------------------------------------------------

    /**
     * Upload a mesh to the GPU and register it.
     * Returns a handle used everywhere else in the engine.
     */
    createMesh(desc: MeshDescriptor): MeshHandle {
        const handle = this._nextHandle++;
        const label  = desc.label ?? `mesh_${handle}`;

        if (desc.vertexLayouts.length !== desc.vertexData.length) {
            throw new Error(
                `[MeshSystem] createMesh "${label}": vertexLayouts.length (${desc.vertexLayouts.length}) ` +
                `!= vertexData.length (${desc.vertexData.length})`
            );
        }

        // --- 1. Vertex buffers ------------------------------------------------
        const vertexBuffers: ResourceHandle[] = [];
        for (let i = 0; i < desc.vertexLayouts.length; i++) {
            const data        = desc.vertexData[i]!;
            const alignedSize = _align4(data.byteLength);

            const vbHandle = this._resources.createBuffer({
                label: `${label}_vb${i}`,
                size:  alignedSize,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
            this._resources.writeBuffer(vbHandle, data);
            vertexBuffers.push(vbHandle);
        }

        // --- 2. Index buffer --------------------------------------------------
        let indexBuffer: ResourceHandle | undefined;
        let indexFormat: GPUIndexFormat = 'uint16';

        if (desc.indexData) {
            indexFormat    = desc.indexData instanceof Uint32Array ? 'uint32' : 'uint16';
            const alignedSize = _align4(desc.indexData.byteLength);

            indexBuffer = this._resources.createBuffer({
                label: `${label}_ib`,
                size:  alignedSize,
                usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            });
            // Cast needed: TypeScript types Uint16/32Array with ArrayBufferLike,
            // but GPUAllowSharedBufferSource requires a concrete ArrayBuffer.
            this._resources.writeBuffer(indexBuffer, desc.indexData as unknown as GPUAllowSharedBufferSource);
        }

        // --- 3. AABB ----------------------------------------------------------
        const aabb = desc.aabb ?? _computeAABBFromDescriptor(desc);

        // --- 4. Bounding sphere -----------------------------------------------
        const boundingSphere = MeshSystem.aabbToSphere(aabb);

        // --- 5. LOD levels ----------------------------------------------------
        const lodLevels = (desc.lodLevels && desc.lodLevels.length > 0)
            ? desc.lodLevels
            : [_defaultLOD(desc)];

        // --- 6. Store record --------------------------------------------------
        const record: MeshRecord = {
            handle,
            label,
            vertexBuffers,
            indexBuffer,
            indexFormat,
            vertexLayouts: desc.vertexLayouts,
            lodLevels,
            aabb,
            boundingSphere,
        };
        this._meshes.set(handle, record);

        this._log.info(
            `Created mesh "${label}" handle=${handle}  ` +
            `VBs=${vertexBuffers.length}  indexed=${!!indexBuffer}  ` +
            `verts≈${_vertexCount(desc)}  ` +
            `AABB=[${_fmtVec(aabb.min)}, ${_fmtVec(aabb.max)}]`
        );
        return handle;
    }

    /**
     * Returns the resolved GPU objects needed to encode a draw call.
     * Returns undefined if the handle is unknown.
     */
    getDrawData(handle: MeshHandle): MeshDrawData | undefined {
        const rec = this._meshes.get(handle);
        if (!rec) return undefined;

        const vertexBuffers: GPUBuffer[] = [];
        for (const vbHandle of rec.vertexBuffers) {
            const buf = this._resources.getBuffer(vbHandle);
            if (!buf) return undefined; // resource was evicted — shouldn't happen for persistent meshes
            vertexBuffers.push(buf);
        }

        const indexBuffer = rec.indexBuffer
            ? (this._resources.getBuffer(rec.indexBuffer) ?? null)
            : null;

        return {
            vertexBuffers,
            vertexLayouts:  rec.vertexLayouts,
            indexBuffer,
            indexFormat:    rec.indexFormat,
            lodLevels:      rec.lodLevels,
            aabb:           rec.aabb,
            boundingSphere: rec.boundingSphere,
        };
    }

    /** @deprecated Use getDrawData() for render-pass use. */
    getMesh(handle: MeshHandle): MeshRecord | undefined {
        return this._meshes.get(handle);
    }

    destroyMesh(handle: MeshHandle): void {
        const rec = this._meshes.get(handle);
        if (!rec) return;

        for (const vbHandle of rec.vertexBuffers) {
            this._resources.destroyBuffer(vbHandle);
        }
        if (rec.indexBuffer !== undefined) {
            this._resources.destroyBuffer(rec.indexBuffer);
        }

        this._meshes.delete(handle);
        this._log.debug(`Destroyed mesh "${rec.label}" handle=${handle}`);
    }

    // -------------------------------------------------------------------------
    // Bounding volumes
    // -------------------------------------------------------------------------

    getAABB(handle: MeshHandle): AABB | undefined {
        return this._meshes.get(handle)?.aabb;
    }

    getBoundingSphere(handle: MeshHandle): BoundingSphere | undefined {
        return this._meshes.get(handle)?.boundingSphere;
    }

    /**
     * Compute an AABB from a tightly-packed Float32Array of xyz positions.
     */
    static computeAABB(positions: Float32Array): AABB {
        if (positions.length < 3) {
            return { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } };
        }

        let minX = positions[0]!, minY = positions[1]!, minZ = positions[2]!;
        let maxX = minX,          maxY = minY,          maxZ = minZ;

        for (let i = 3; i < positions.length; i += 3) {
            const x = positions[i]!, y = positions[i + 1]!, z = positions[i + 2]!;
            if (x < minX) minX = x; else if (x > maxX) maxX = x;
            if (y < minY) minY = y; else if (y > maxY) maxY = y;
            if (z < minZ) minZ = z; else if (z > maxZ) maxZ = z;
        }

        return {
            min: { x: minX, y: minY, z: minZ },
            max: { x: maxX, y: maxY, z: maxZ },
        };
    }

    /**
     * Derive a bounding sphere from an AABB.
     * center = (min + max) / 2,  radius = |max - center|
     */
    static aabbToSphere(aabb: AABB): BoundingSphere {
        const cx = (aabb.min.x + aabb.max.x) * 0.5;
        const cy = (aabb.min.y + aabb.max.y) * 0.5;
        const cz = (aabb.min.z + aabb.max.z) * 0.5;

        const dx = aabb.max.x - cx;
        const dy = aabb.max.y - cy;
        const dz = aabb.max.z - cz;

        return {
            center: { x: cx, y: cy, z: cz },
            radius: Math.sqrt(dx * dx + dy * dy + dz * dz),
        };
    }

    // -------------------------------------------------------------------------
    // LOD selection
    // -------------------------------------------------------------------------

    /**
     * Return the appropriate LOD index based on screen-space coverage (0–1).
     * LOD 0 is highest detail (coverage → 1). Returns 0 if no LODs defined.
     */
    selectLOD(handle: MeshHandle, screenCoverage: number): number {
        const mesh = this._meshes.get(handle);
        if (!mesh || mesh.lodLevels.length === 0) return 0;

        // Walk from lowest-detail to highest, pick first that qualifies
        for (let i = mesh.lodLevels.length - 1; i >= 0; i--) {
            if (screenCoverage <= mesh.lodLevels[i]!.screenCoverage) return i;
        }
        return 0;
    }

    // -------------------------------------------------------------------------
    // Future: RT acceleration structures
    // -------------------------------------------------------------------------

    /**
     * Build a BLAS for the given mesh. Requires FeatureTier.Tier2_RT.
     */
    buildBLAS(_handle: MeshHandle): void {
        // TODO: create bottom-level acceleration structure from vertex/index data
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    destroy(): void {
        for (const [handle] of this._meshes) {
            this.destroyMesh(handle);
        }
    }
}

// -------------------------------------------------------------------------
// Private helpers
// -------------------------------------------------------------------------

/** Round byte size up to the nearest 4-byte multiple (WebGPU alignment rule). */
function _align4(byteSize: number): number {
    return (byteSize + 3) & ~3;
}

/**
 * Extract positions from a MeshDescriptor by finding the POSITION attribute
 * and striding through the interleaved (or split) vertex buffer.
 */
function _computeAABBFromDescriptor(desc: MeshDescriptor): AABB {
    for (let bi = 0; bi < desc.vertexLayouts.length; bi++) {
        const layout = desc.vertexLayouts[bi]!;
        const posAttr = layout.attributes.find(a => a.semantic === VertexSemantic.Position);
        if (!posAttr) continue;

        const data    = desc.vertexData[bi]!;
        const stride  = layout.arrayStride;
        const offset  = posAttr.offset;
        const count   = Math.floor(data.byteLength / stride);
        const view    = new DataView(data);
        const packed  = new Float32Array(count * 3);

        for (let i = 0; i < count; i++) {
            const base = i * stride + offset;
            packed[i * 3 + 0] = view.getFloat32(base + 0, true);
            packed[i * 3 + 1] = view.getFloat32(base + 4, true);
            packed[i * 3 + 2] = view.getFloat32(base + 8, true);
        }

        return MeshSystem.computeAABB(packed);
    }

    return { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } };
}

/** Build a single-LOD, single-SubMesh default when none is supplied. */
function _defaultLOD(desc: MeshDescriptor): LODLevel {
    const indexCount  = desc.indexData?.length ?? 0;
    const vertexCount = desc.vertexLayouts[0] && desc.vertexData[0]
        ? Math.floor(desc.vertexData[0].byteLength / desc.vertexLayouts[0].arrayStride)
        : 0;

    return {
        screenCoverage: 1.0,
        subMeshes: [{
            indexOffset:  0,
            indexCount,
            vertexOffset: 0,
            vertexCount,
            materialIndex: 0,
        }],
    };
}

/** Approximate vertex count from the first vertex buffer. */
function _vertexCount(desc: MeshDescriptor): number {
    if (!desc.vertexLayouts[0] || !desc.vertexData[0]) return 0;
    return Math.floor(desc.vertexData[0].byteLength / desc.vertexLayouts[0].arrayStride);
}

function _fmtVec(v: Vec3): string {
    return `${v.x.toFixed(2)},${v.y.toFixed(2)},${v.z.toFixed(2)}`;
}
