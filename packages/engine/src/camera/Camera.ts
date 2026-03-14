// /src/engine/camera/Camera.ts

import { mat4 } from 'gl-matrix';
import type { Mat4, NodeHandle } from '../scene/SceneGraph';

// -------------------------------------------------------------------------
// Types
// -------------------------------------------------------------------------

export type CameraHandle = number;

export enum ProjectionType {
    Perspective  = 'PERSPECTIVE',
    Orthographic = 'ORTHOGRAPHIC',
}

export interface PerspectiveParams {
    fovY: number;      // radians
    aspectRatio: number;
    near: number;
    far: number;
}

export interface OrthographicParams {
    left: number;
    right: number;
    bottom: number;
    top: number;
    near: number;
    far: number;
}

/**
 * Six frustum planes in world space (used for culling).
 * Each plane is [a, b, c, d] where ax + by + cz + d >= 0 means inside.
 */
export type FrustumPlanes = [Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array];

export interface CameraDescriptor {
    label?: string;
    projectionType?: ProjectionType;
    perspective?: Partial<PerspectiveParams>;
    orthographic?: Partial<OrthographicParams>;
    /** Scene node this camera is attached to (inherits world transform). */
    nodeHandle?: NodeHandle;
    /** TAA jitter enabled by default */
    taaEnabled?: boolean;
    /** Exposure value (EV100). */
    exposure?: number;
}

/**
 * Internal camera record.
 */
export interface CameraRecord {
    handle: CameraHandle;
    label: string;
    projectionType: ProjectionType;
    perspective: PerspectiveParams;
    orthographic: OrthographicParams;
    nodeHandle: NodeHandle | null;
    taaEnabled: boolean;
    exposure: number;

    // Cached matrices (column-major Float32Array, compatible with gl-matrix)
    viewMatrix: Mat4;
    projectionMatrix: Mat4;
    viewProjectionMatrix: Mat4;
    inverseViewProjection: Mat4;
    frustumPlanes: FrustumPlanes;

    // TAA state
    jitterX: number;   // sub-pixel offset in [-0.5, 0.5] pixel space
    jitterY: number;
    taaFrameIndex: number;
}

/**
 * Manages cameras: builds view/projection matrices, extracts frustum planes,
 * applies TAA sub-pixel jitter, and provides exposure settings for tonemapping.
 */
export class CameraSystem {

    private _nextHandle: CameraHandle = 1;
    private _cameras: Map<CameraHandle, CameraRecord> = new Map();
    private _activeCameraHandle: CameraHandle | null = null;

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    init(): void {
        // Pure CPU-side; no GPU dependency
    }

    // -------------------------------------------------------------------------
    // Camera CRUD
    // -------------------------------------------------------------------------

    createCamera(desc: CameraDescriptor): CameraHandle {
        const handle = this._nextHandle++;

        const perspective: PerspectiveParams = {
            fovY:        desc.perspective?.fovY        ?? Math.PI / 3,   // 60°
            aspectRatio: desc.perspective?.aspectRatio ?? 16 / 9,
            near:        desc.perspective?.near        ?? 0.1,
            far:         desc.perspective?.far         ?? 1000,
        };

        const orthographic: OrthographicParams = {
            left:   desc.orthographic?.left   ?? -10,
            right:  desc.orthographic?.right  ??  10,
            bottom: desc.orthographic?.bottom ?? -10,
            top:    desc.orthographic?.top    ??  10,
            near:   desc.orthographic?.near   ?? 0.1,
            far:    desc.orthographic?.far    ?? 100,
        };

        // mat4.create() returns a Float32Array(16) identity matrix at runtime.
        // Cast through unknown because gl-matrix's return type is 'mat4', not Float32Array.
        const newMat4 = (): Mat4 => mat4.create() as unknown as Float32Array;

        const record: CameraRecord = {
            handle,
            label:          desc.label ?? `camera_${handle}`,
            projectionType: desc.projectionType ?? ProjectionType.Perspective,
            perspective,
            orthographic,
            nodeHandle:     desc.nodeHandle ?? null,
            taaEnabled:     desc.taaEnabled ?? true,
            exposure:       desc.exposure   ?? 0,

            viewMatrix:            newMat4(),
            projectionMatrix:      newMat4(),
            viewProjectionMatrix:  newMat4(),
            inverseViewProjection: newMat4(),

            frustumPlanes: [
                new Float32Array(4), new Float32Array(4), new Float32Array(4),
                new Float32Array(4), new Float32Array(4), new Float32Array(4),
            ] as FrustumPlanes,

            jitterX: 0,
            jitterY: 0,
            taaFrameIndex: 0,
        };

        this._cameras.set(handle, record);
        return handle;
    }

    getCamera(handle: CameraHandle): CameraRecord | undefined {
        return this._cameras.get(handle);
    }

    destroyCamera(handle: CameraHandle): void {
        this._cameras.delete(handle);
        if (this._activeCameraHandle === handle) this._activeCameraHandle = null;
    }

    setActiveCamera(handle: CameraHandle): void {
        this._activeCameraHandle = handle;
    }

    getActiveCamera(): CameraRecord | undefined {
        if (this._activeCameraHandle === null) return undefined;
        return this._cameras.get(this._activeCameraHandle);
    }

    // -------------------------------------------------------------------------
    // Matrix computation
    // -------------------------------------------------------------------------

    /**
     * Recompute view, projection, VP, and frustum planes for a camera.
     * Call after scene graph world matrices are updated.
     *
     * @param worldMatrix  World matrix of the scene node this camera is attached to.
     *                     The view matrix is the inverse of this.
     */
    update(handle: CameraHandle, worldMatrix: Mat4): void {
        const cam = this._cameras.get(handle);
        if (!cam) return;

        // viewMatrix = inverse(worldMatrix)
        // Falls back to identity if the matrix is degenerate.
        if (!mat4.invert(cam.viewMatrix, worldMatrix)) {
            mat4.identity(cam.viewMatrix);
        }

        // projectionMatrix — use ZO variants so Z maps to [0, 1] (WebGPU NDC).
        if (cam.projectionType === ProjectionType.Perspective) {
            const { fovY, aspectRatio, near, far } = cam.perspective;
            mat4.perspectiveZO(cam.projectionMatrix, fovY, aspectRatio, near, far);
        } else {
            const { left, right, bottom, top, near, far } = cam.orthographic;
            mat4.orthoZO(cam.projectionMatrix, left, right, bottom, top, near, far);
        }

        // viewProjectionMatrix = projection * view
        mat4.multiply(cam.viewProjectionMatrix, cam.projectionMatrix, cam.viewMatrix);

        // inverseViewProjection = inverse(VP)  — used for ray-casting / light volumes
        if (!mat4.invert(cam.inverseViewProjection, cam.viewProjectionMatrix)) {
            mat4.identity(cam.inverseViewProjection);
        }

        // Frustum planes from the VP matrix (WebGPU [0,1] Z range)
        cam.frustumPlanes = CameraSystem.extractFrustumPlanes(cam.viewProjectionMatrix);
    }

    /**
     * Rebuild projection matrix (call after aspect ratio / FOV change).
     */
    updateProjection(handle: CameraHandle, aspectRatio: number): void {
        const cam = this._cameras.get(handle);
        if (!cam) return;
        cam.perspective.aspectRatio = aspectRatio;
        // Recompute projection; view stays the same — call update() next frame
        if (cam.projectionType === ProjectionType.Perspective) {
            const { fovY, near, far } = cam.perspective;
            mat4.perspectiveZO(cam.projectionMatrix, fovY, aspectRatio, near, far);
        }
        mat4.multiply(cam.viewProjectionMatrix, cam.projectionMatrix, cam.viewMatrix);
        if (!mat4.invert(cam.inverseViewProjection, cam.viewProjectionMatrix)) {
            mat4.identity(cam.inverseViewProjection);
        }
    }

    // -------------------------------------------------------------------------
    // TAA jitter
    // -------------------------------------------------------------------------

    /**
     * Advance TAA frame counter and compute the next Halton sub-pixel jitter.
     *
     * Stores jitterX / jitterY as sub-pixel offsets in [-0.5, 0.5] pixel space.
     * Callers (e.g. FrameOrchestrator) scale to NDC before writing the uniform
     * buffer: `jitter_ndc = jitter_pixel / resolution * 2`.
     *
     * The jitter sequence repeats every 64 frames (sufficient for most TAA needs).
     */
    advanceTAAJitter(handle: CameraHandle): void {
        const cam = this._cameras.get(handle);
        if (!cam || !cam.taaEnabled) return;

        cam.taaFrameIndex = (cam.taaFrameIndex + 1) % 64;
        cam.jitterX = _halton(2, cam.taaFrameIndex) - 0.5;
        cam.jitterY = _halton(3, cam.taaFrameIndex) - 0.5;
    }

    // -------------------------------------------------------------------------
    // Frustum extraction
    // -------------------------------------------------------------------------

    /**
     * Extract 6 frustum planes from a view-projection matrix using the
     * Gribb-Hartmann method, adapted for WebGPU's [0, 1] NDC Z range.
     *
     * Each plane [a, b, c, d]: a point P is inside when a·Px + b·Py + c·Pz + d ≥ 0.
     *
     * The matrix is assumed column-major (gl-matrix convention):
     *   m[row + col*4] = element at (row, col)
     */
    static extractFrustumPlanes(vp: Mat4): FrustumPlanes {
        const m = vp as Float32Array;

        // Row i of the matrix: [m[i], m[i+4], m[i+8], m[i+12]]
        const planes: FrustumPlanes = [
            new Float32Array(4), // left
            new Float32Array(4), // right
            new Float32Array(4), // bottom
            new Float32Array(4), // top
            new Float32Array(4), // near
            new Float32Array(4), // far
        ] as FrustumPlanes;

        // Left:   row3 + row0
        planes[0][0] = m[3] + m[0];  planes[0][1] = m[7] + m[4];
        planes[0][2] = m[11] + m[8]; planes[0][3] = m[15] + m[12];

        // Right:  row3 - row0
        planes[1][0] = m[3] - m[0];  planes[1][1] = m[7] - m[4];
        planes[1][2] = m[11] - m[8]; planes[1][3] = m[15] - m[12];

        // Bottom: row3 + row1
        planes[2][0] = m[3] + m[1];  planes[2][1] = m[7] + m[5];
        planes[2][2] = m[11] + m[9]; planes[2][3] = m[15] + m[13];

        // Top:    row3 - row1
        planes[3][0] = m[3] - m[1];  planes[3][1] = m[7] - m[5];
        planes[3][2] = m[11] - m[9]; planes[3][3] = m[15] - m[13];

        // Near:   row2        (WebGPU Z ∈ [0,1]: near at clip.z = 0)
        planes[4][0] = m[2];  planes[4][1] = m[6];
        planes[4][2] = m[10]; planes[4][3] = m[14];

        // Far:    row3 - row2 (WebGPU Z ∈ [0,1]: far at clip.z = clip.w)
        planes[5][0] = m[3] - m[2];  planes[5][1] = m[7] - m[6];
        planes[5][2] = m[11] - m[10]; planes[5][3] = m[15] - m[14];

        // Normalise all planes so the signed distance is in world-space units
        for (const p of planes) {
            const len = Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            if (len > 1e-6) { p[0] /= len; p[1] /= len; p[2] /= len; p[3] /= len; }
        }

        return planes;
    }

    // -------------------------------------------------------------------------
    // Exposure
    // -------------------------------------------------------------------------

    getExposure(handle: CameraHandle): number {
        return this._cameras.get(handle)?.exposure ?? 0;
    }

    setExposure(handle: CameraHandle, ev100: number): void {
        const cam = this._cameras.get(handle);
        if (cam) cam.exposure = ev100;
    }

    /**
     * Convert EV100 to a linear exposure multiplier for the tonemapper.
     * Based on the Lagarde/de Rousiers physically-based camera model.
     */
    static ev100ToExposure(ev100: number): number {
        return 1.0 / (Math.pow(2.0, ev100) * 1.2);
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    destroy(): void {
        this._cameras.clear();
        this._activeCameraHandle = null;
    }
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

/**
 * Halton low-discrepancy sequence.
 * Returns a value in [0, 1) for the given base and index.
 */
function _halton(base: number, index: number): number {
    let result = 0;
    let f = 1;
    let i = index;
    while (i > 0) {
        f /= base;
        result += f * (i % base);
        i = Math.floor(i / base);
    }
    return result;
}

