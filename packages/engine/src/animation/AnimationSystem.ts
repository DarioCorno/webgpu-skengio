// /src/engine/animation/AnimationSystem.ts
//
// Manages skeletal animation clips, skins, and per-frame playback evaluation.
//
// Responsibilities:
//   - Store animation clips (keyframe data parsed from glTF or created manually)
//   - Store skin definitions (joint hierarchies + inverse bind matrices)
//   - Manage playback state per target (which clips are playing, time, speed, loop)
//   - Each frame: evaluate keyframes -> update node transforms -> compute joint matrices
//   - Provide joint matrices for the renderer (GPU storage buffers)
//
// TODO: Morph target / blend shape animation
// TODO: Animation blending and crossfade
// TODO: Animation events / callbacks
// TODO: Additive animation layers

import { mat4, quat, vec3 } from 'gl-matrix';
import { Logger } from '../core/Logger';
import type { GPUBackend } from '../core/GPUBackend';
import type { SceneGraph, NodeHandle, Transform } from '../scene/SceneGraph';
import type { GLTFAnimationData, GLTFChannelData, GLTFSkinData } from '../scene/GLTFLoader';

// -------------------------------------------------------------------------
// Public types
// -------------------------------------------------------------------------

export type AnimClipHandle = number;
export type SkinHandle = number;

/** Playback configuration for an animation. */
export interface PlaybackConfig {
    /** Clip to play — handle or name. */
    clip: AnimClipHandle | string;
    /** Playback speed multiplier (default: 1.0). */
    speed?: number;
    /** Loop the animation (default: true). */
    loop?: boolean;
    /** Start playing immediately (default: true). */
    autoplay?: boolean;
}

/** Read-only view of the current playback state. */
export interface PlaybackState {
    clipHandle: AnimClipHandle;
    clipName: string;
    time: number;
    speed: number;
    loop: boolean;
    playing: boolean;
    duration: number;
}

// -------------------------------------------------------------------------
// Internal types
// -------------------------------------------------------------------------

interface AnimationClipRecord {
    handle: AnimClipHandle;
    name: string;
    duration: number;
    channels: GLTFChannelData[];
}

interface SkinRecord {
    handle: SkinHandle;
    name: string;
    joints: NodeHandle[];
    inverseBindMatrices: Float32Array;  // N * 16 floats
    jointCount: number;
    /** CPU-side joint matrices — recomputed each frame. */
    jointMatrices: Float32Array;        // N * 16 floats
    /** The mesh node this skin is attached to. */
    meshNode: NodeHandle;
    /** GPU storage buffer holding joint matrices (N * 64 bytes). */
    gpuBuffer: GPUBuffer;
    dirty: boolean;
}

interface ActivePlayback {
    clipHandle: AnimClipHandle;
    time: number;
    speed: number;
    loop: boolean;
    playing: boolean;
}

// -------------------------------------------------------------------------
// AnimationSystem
// -------------------------------------------------------------------------

export class AnimationSystem {

    private readonly _log = new Logger('AnimationSystem');

    private _backend!: GPUBackend;
    private _sceneGraph!: SceneGraph;

    private _nextClipHandle: AnimClipHandle = 1;
    private _nextSkinHandle: SkinHandle = 1;

    private _clips: Map<AnimClipHandle, AnimationClipRecord> = new Map();
    private _clipsByName: Map<string, AnimClipHandle> = new Map();
    private _skins: Map<SkinHandle, SkinRecord> = new Map();

    /** Map from mesh node handle -> skin handle. */
    private _nodeSkinMap: Map<NodeHandle, SkinHandle> = new Map();
    /** Map from mesh node handle -> active playback. */
    private _nodePlaybacks: Map<NodeHandle, ActivePlayback> = new Map();

    // Temp buffers for interpolation (avoid allocation per frame)
    private _tmpVec3A = vec3.create();
    private _tmpVec3B = vec3.create();
    private _tmpQuatA = quat.create();
    private _tmpQuatB = quat.create();
    private _tmpMat4  = mat4.create();

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    init(backend: GPUBackend, sceneGraph: SceneGraph): void {
        this._backend = backend;
        this._sceneGraph = sceneGraph;
    }

    // -------------------------------------------------------------------------
    // Clip registration
    // -------------------------------------------------------------------------

    /**
     * Register an animation clip (typically parsed from glTF).
     * Returns a handle for later playback reference.
     */
    registerClip(data: GLTFAnimationData): AnimClipHandle {
        const handle = this._nextClipHandle++;
        const record: AnimationClipRecord = {
            handle,
            name: data.name,
            duration: data.duration,
            channels: data.channels,
        };
        this._clips.set(handle, record);
        this._clipsByName.set(data.name, handle);
        this._log.info(`Registered clip "${data.name}" handle=${handle} duration=${data.duration.toFixed(2)}s channels=${data.channels.length}`);
        return handle;
    }

    /** Look up a clip handle by name. */
    getClipByName(name: string): AnimClipHandle | undefined {
        return this._clipsByName.get(name);
    }

    /** List all registered clip names. */
    getClipNames(): string[] {
        return [...this._clipsByName.keys()];
    }

    // -------------------------------------------------------------------------
    // Skin registration
    // -------------------------------------------------------------------------

    /**
     * Register a skin (skeleton) definition and allocate its GPU joint buffer.
     * @param data      Skin data from GLTFLoader.
     * @param meshNode  The scene-graph node the skinned mesh is attached to.
     */
    registerSkin(data: GLTFSkinData, meshNode: NodeHandle): SkinHandle {
        const handle = this._nextSkinHandle++;
        const jointCount = data.jointNodeHandles.length;

        const gpuBuffer = this._backend.device.createBuffer({
            label: `Skin_${data.name}_joints`,
            size: Math.max(jointCount * 64, 64), // at least 64 bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const record: SkinRecord = {
            handle,
            name: data.name,
            joints: data.jointNodeHandles,
            inverseBindMatrices: data.inverseBindMatrices,
            jointCount,
            jointMatrices: new Float32Array(jointCount * 16),
            meshNode,
            gpuBuffer,
            dirty: true,
        };

        this._skins.set(handle, record);
        this._nodeSkinMap.set(meshNode, handle);

        this._log.info(`Registered skin "${data.name}" handle=${handle} joints=${jointCount} meshNode=${meshNode}`);
        return handle;
    }

    /** Get the skin handle attached to a mesh node (if any). */
    getSkinForNode(nodeHandle: NodeHandle): SkinHandle | undefined {
        return this._nodeSkinMap.get(nodeHandle);
    }

    /** Get the GPU joint matrix buffer for a skin. */
    getJointBuffer(skinHandle: SkinHandle): GPUBuffer | null {
        return this._skins.get(skinHandle)?.gpuBuffer ?? null;
    }

    /** Get joint count for a skin. */
    getJointCount(skinHandle: SkinHandle): number {
        return this._skins.get(skinHandle)?.jointCount ?? 0;
    }

    // -------------------------------------------------------------------------
    // Playback control
    // -------------------------------------------------------------------------

    /**
     * Start playing an animation clip on a node (and its skeleton).
     * The node should be the root of a skinned mesh hierarchy.
     */
    play(nodeHandle: NodeHandle, config: PlaybackConfig): void {
        const clipHandle = typeof config.clip === 'string'
            ? this._clipsByName.get(config.clip)
            : config.clip;

        if (clipHandle === undefined) {
            this._log.warn(`Cannot play clip "${config.clip}" — not found`);
            return;
        }

        const clip = this._clips.get(clipHandle);
        if (!clip) {
            this._log.warn(`Cannot play clip handle=${clipHandle} — not registered`);
            return;
        }

        this._nodePlaybacks.set(nodeHandle, {
            clipHandle,
            time: 0,
            speed: config.speed ?? 1.0,
            loop: config.loop ?? true,
            playing: config.autoplay !== false,
        });
    }

    /** Stop playback on a node. */
    stop(nodeHandle: NodeHandle): void {
        const pb = this._nodePlaybacks.get(nodeHandle);
        if (pb) pb.playing = false;
    }

    /** Resume playback on a node. */
    resume(nodeHandle: NodeHandle): void {
        const pb = this._nodePlaybacks.get(nodeHandle);
        if (pb) pb.playing = true;
    }

    /** Get current playback state for a node. */
    getPlaybackState(nodeHandle: NodeHandle): PlaybackState | undefined {
        const pb = this._nodePlaybacks.get(nodeHandle);
        if (!pb) return undefined;
        const clip = this._clips.get(pb.clipHandle);
        if (!clip) return undefined;
        return {
            clipHandle: pb.clipHandle,
            clipName: clip.name,
            time: pb.time,
            speed: pb.speed,
            loop: pb.loop,
            playing: pb.playing,
            duration: clip.duration,
        };
    }

    // -------------------------------------------------------------------------
    // Per-frame update
    // -------------------------------------------------------------------------

    /**
     * Advance all active animations by `dt` seconds, evaluate keyframes,
     * and update scene-graph node transforms.
     *
     * Call this BEFORE SceneGraph.propagateTransforms() so that the updated
     * local transforms get propagated to world matrices.
     */
    updateAnimations(dt: number): void {
        for (const [_nodeHandle, pb] of this._nodePlaybacks) {
            if (!pb.playing) continue;

            const clip = this._clips.get(pb.clipHandle);
            if (!clip || clip.duration <= 0) continue;

            // Advance time
            pb.time += dt * pb.speed;

            if (pb.loop) {
                pb.time = pb.time % clip.duration;
                if (pb.time < 0) pb.time += clip.duration;
            } else {
                if (pb.time >= clip.duration) {
                    pb.time = clip.duration;
                    pb.playing = false;
                } else if (pb.time < 0) {
                    pb.time = 0;
                    pb.playing = false;
                }
            }

            // Evaluate all channels at current time
            this._evaluateClip(clip, pb.time);
        }
    }

    /**
     * Compute joint matrices for all registered skins using the current
     * scene-graph world matrices.
     *
     * Call this AFTER SceneGraph.propagateTransforms() so that joint world
     * matrices are up to date.
     */
    computeJointMatrices(): void {
        const inverseMeshWorld = this._tmpMat4;

        for (const skin of this._skins.values()) {
            const meshWorldMatrix = this._sceneGraph.getWorldMatrix(skin.meshNode);
            if (!meshWorldMatrix) continue;

            mat4.invert(inverseMeshWorld, meshWorldMatrix);

            const ibm = skin.inverseBindMatrices;
            const out = skin.jointMatrices;

            for (let j = 0; j < skin.jointCount; j++) {
                const jointNode = skin.joints[j]!;
                const jointWorld = this._sceneGraph.getWorldMatrix(jointNode);
                if (!jointWorld) {
                    mat4.identity(out.subarray(j * 16, j * 16 + 16) as unknown as mat4);
                    continue;
                }

                // jointMatrix[j] = inverse(meshWorld) * jointWorld * IBM[j]
                const ibmSlice = ibm.subarray(j * 16, j * 16 + 16);
                const outSlice = out.subarray(j * 16, j * 16 + 16);

                mat4.multiply(
                    outSlice as unknown as mat4,
                    inverseMeshWorld as unknown as mat4,
                    jointWorld as unknown as mat4,
                );
                mat4.multiply(
                    outSlice as unknown as mat4,
                    outSlice as unknown as mat4,
                    ibmSlice as unknown as mat4,
                );
            }

            skin.dirty = true;
        }
    }

    /**
     * Upload dirty joint matrices to the GPU.
     * Call this after computeJointMatrices() and before rendering.
     */
    uploadJointMatrices(): void {
        for (const skin of this._skins.values()) {
            if (skin.dirty) {
                this._backend.queue.writeBuffer(skin.gpuBuffer, 0, skin.jointMatrices as unknown as Float32Array<ArrayBuffer>);
                skin.dirty = false;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Keyframe evaluation
    // -------------------------------------------------------------------------

    private _evaluateClip(clip: AnimationClipRecord, time: number): void {
        // Group channels by target node so we can batch-apply TRS
        // (each node may have up to 3 channels: T, R, S)
        const nodeUpdates = new Map<NodeHandle, {
            translation?: Float32Array,
            rotation?: Float32Array,
            scale?: Float32Array,
        }>();

        for (const channel of clip.channels) {
            const value = this._sampleChannel(channel, time);
            if (!value) continue;

            let entry = nodeUpdates.get(channel.targetNodeHandle);
            if (!entry) {
                entry = {};
                nodeUpdates.set(channel.targetNodeHandle, entry);
            }

            switch (channel.path) {
                case 'translation': entry.translation = value; break;
                case 'rotation':    entry.rotation = value; break;
                case 'scale':       entry.scale = value; break;
                // TODO: 'weights' for morph targets
            }
        }

        // Apply accumulated TRS updates to scene graph
        for (const [nodeHandle, update] of nodeUpdates) {
            const currentTransform = this._sceneGraph.getLocalTransform(nodeHandle);
            if (!currentTransform) continue;

            const newTransform: Transform = {
                position: update.translation
                    ? new Float32Array(update.translation)
                    : currentTransform.position,
                rotation: update.rotation
                    ? new Float32Array(update.rotation)
                    : currentTransform.rotation,
                scale: update.scale
                    ? new Float32Array(update.scale)
                    : currentTransform.scale,
            };

            this._sceneGraph.setLocalTransform(nodeHandle, newTransform);
        }
    }

    /**
     * Sample a single animation channel at the given time.
     * Returns the interpolated value as a Float32Array, or null if empty.
     */
    private _sampleChannel(channel: GLTFChannelData, time: number): Float32Array | null {
        const times = channel.inputTimes;
        const values = channel.outputValues;
        if (times.length === 0) return null;

        // Determine component count from path
        const componentCount = channel.path === 'rotation' ? 4 : 3;

        // Clamp to range
        if (time <= times[0]!) {
            return values.subarray(0, componentCount);
        }
        if (time >= times[times.length - 1]!) {
            const lastIdx = (times.length - 1) * componentCount;
            return values.subarray(lastIdx, lastIdx + componentCount);
        }

        // Binary search for the bracketing keyframes
        let lo = 0, hi = times.length - 1;
        while (lo < hi - 1) {
            const mid = (lo + hi) >> 1;
            if (times[mid]! <= time) lo = mid;
            else hi = mid;
        }

        const t0 = times[lo]!;
        const t1 = times[hi]!;
        const dt = t1 - t0;
        const t = dt > 0 ? (time - t0) / dt : 0;

        switch (channel.interpolation) {
            case 'STEP':
                return values.subarray(lo * componentCount, lo * componentCount + componentCount);

            case 'LINEAR':
                return this._lerpChannel(channel.path, values, lo, hi, componentCount, t);

            case 'CUBICSPLINE':
                return this._cubicSplineChannel(channel.path, values, lo, hi, componentCount, t, dt);

            default:
                return this._lerpChannel(channel.path, values, lo, hi, componentCount, t);
        }
    }

    private _lerpChannel(
        path: string,
        values: Float32Array,
        lo: number,
        hi: number,
        componentCount: number,
        t: number,
    ): Float32Array {
        const offset0 = lo * componentCount;
        const offset1 = hi * componentCount;

        if (path === 'rotation') {
            // Spherical linear interpolation for quaternions
            const q0 = this._tmpQuatA;
            const q1 = this._tmpQuatB;
            q0[0] = values[offset0]!;     q0[1] = values[offset0 + 1]!;
            q0[2] = values[offset0 + 2]!; q0[3] = values[offset0 + 3]!;
            q1[0] = values[offset1]!;     q1[1] = values[offset1 + 1]!;
            q1[2] = values[offset1 + 2]!; q1[3] = values[offset1 + 3]!;
            quat.slerp(q0, q0, q1, t);
            return new Float32Array([q0[0], q0[1], q0[2], q0[3]]);
        }

        // Linear interpolation for translation/scale
        const v0 = this._tmpVec3A;
        const v1 = this._tmpVec3B;
        v0[0] = values[offset0]!;     v0[1] = values[offset0 + 1]!; v0[2] = values[offset0 + 2]!;
        v1[0] = values[offset1]!;     v1[1] = values[offset1 + 1]!; v1[2] = values[offset1 + 2]!;
        vec3.lerp(v0, v0, v1, t);
        return new Float32Array([v0[0], v0[1], v0[2]]);
    }

    private _cubicSplineChannel(
        path: string,
        values: Float32Array,
        lo: number,
        hi: number,
        componentCount: number,
        t: number,
        dt: number,
    ): Float32Array {
        // glTF cubicspline: each keyframe stores [inTangent, value, outTangent]
        // so stride = componentCount * 3
        const stride = componentCount * 3;
        const v0Offset  = lo * stride + componentCount;        // value at lo
        const outOffset = lo * stride + componentCount * 2;    // out-tangent at lo
        const inOffset  = hi * stride;                          // in-tangent at hi
        const v1Offset  = hi * stride + componentCount;        // value at hi

        const t2 = t * t;
        const t3 = t2 * t;

        // Hermite basis functions
        const h00 = 2 * t3 - 3 * t2 + 1;
        const h10 = t3 - 2 * t2 + t;
        const h01 = -2 * t3 + 3 * t2;
        const h11 = t3 - t2;

        const result = new Float32Array(componentCount);
        for (let c = 0; c < componentCount; c++) {
            const p0 = values[v0Offset + c]!;
            const m0 = values[outOffset + c]! * dt;
            const p1 = values[v1Offset + c]!;
            const m1 = values[inOffset + c]! * dt;
            result[c] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
        }

        // Normalize quaternions after cubic interpolation
        if (path === 'rotation') {
            const len = Math.sqrt(result[0]! * result[0]! + result[1]! * result[1]! +
                                  result[2]! * result[2]! + result[3]! * result[3]!);
            if (len > 1e-10) {
                result[0]! /= len; result[1]! /= len;
                result[2]! /= len; result[3]! /= len;
            }
        }

        return result;
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    destroy(): void {
        for (const skin of this._skins.values()) {
            skin.gpuBuffer.destroy();
        }
        this._clips.clear();
        this._clipsByName.clear();
        this._skins.clear();
        this._nodeSkinMap.clear();
        this._nodePlaybacks.clear();
    }
}
