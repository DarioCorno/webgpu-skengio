// /src/engine/input/CameraController.ts
//
// Camera controller interface and built-in implementations:
//
//   FreeLookController  — WASD movement + mouse-look (default)
//   CameraControllerSystem — registry + enable/disable management

import { quat, vec3 } from 'gl-matrix';
import type { InputState, InputSystem } from './InputSystem';
import type { SceneGraph, NodeHandle } from '../scene/SceneGraph';

// ─────────────────────────────────────────────────────────────────────────────
// Input device selection
// ─────────────────────────────────────────────────────────────────────────────

export type InputDevice = 'mouse' | 'gamepad';

// ─────────────────────────────────────────────────────────────────────────────
// CameraController interface
// ─────────────────────────────────────────────────────────────────────────────

export interface CameraController {
    /** Unique name used to select this controller in the system. */
    readonly name: string;

    /** Toggle the controller without removing it from the registry. */
    enabled: boolean;

    /** Which input device drives this controller. */
    device: InputDevice;

    /** Invert horizontal look/orbit axis. */
    invertX: boolean;

    /** Invert vertical look/orbit axis. */
    invertY: boolean;

    /**
     * Whether this controller needs pointer lock (cursor hidden, infinite drag).
     * Default: true.  Orbit-style controllers set this to false.
     */
    readonly wantsPointerLock?: boolean;

    /** Called once just before this controller becomes active. */
    onActivate?(canvas: HTMLCanvasElement): void;

    /** Called once just after this controller stops being active. */
    onDeactivate?(): void;

    /**
     * Apply camera movement for this frame.
     * @param dt          Delta time in seconds.
     * @param input       Current frame's raw input state.
     * @param scene       Scene graph to write the updated transform into.
     * @param cameraNode  Handle of the camera scene node to move.
     */
    update(dt: number, input: InputState, scene: SceneGraph, cameraNode: NodeHandle): void;

    /** Release any resources held by this controller. */
    destroy?(): void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Build a quaternion from separate yaw (world-Y) and pitch (local-X) angles. */
function _quatFromYawPitch(yaw: number, pitch: number): quat {
    const yawQ   = quat.create();
    const pitchQ = quat.create();
    quat.setAxisAngle(yawQ,   [0, 1, 0], yaw);
    quat.setAxisAngle(pitchQ, [1, 0, 0], pitch);
    const out = quat.create();
    quat.multiply(out, yawQ, pitchQ);
    quat.normalize(out, out);
    return out;
}

/**
 * Extract yaw (world-Y) and pitch (local-X) from a quaternion built via
 * _quatFromYawPitch (i.e. Yaw * Pitch order = Y-then-X intrinsic).
 */
function _yawPitchFromQuat(q: Float32Array): { yaw: number; pitch: number } {
    const [x, y, z, w] = q;
    // Pitch (rotation around X) — from Tait-Bryan YXZ decomposition
    const sinP = 2 * (w * x - y * z);
    const pitch = Math.abs(sinP) >= 1
        ? Math.sign(sinP) * HALF_PI
        : Math.asin(sinP);
    // Yaw (rotation around Y)
    const sinY = 2 * (w * y + x * z);
    const cosY = 1 - 2 * (x * x + y * y);
    const yaw = Math.atan2(sinY, cosY);
    return { yaw, pitch };
}

/** Write position + rotation back into a scene node's local transform. */
function _applyTransform(scene: SceneGraph, node: NodeHandle, pos: vec3, rotation: quat): void {
    scene.setLocalTransform(node, {
        position: new Float32Array([pos[0]!, pos[1]!, pos[2]!]),
        rotation: new Float32Array([rotation[0]!, rotation[1]!, rotation[2]!, rotation[3]!]),
    });
}

const HALF_PI = Math.PI / 2 - 0.001; // pitch clamp (avoids gimbal flip)

/**
 * Build a quaternion from an orthonormal basis (right, up, forward).
 * The basis maps to the camera convention: +X=right, +Y=up, -Z=forward.
 * So the rotation matrix columns are: right, up, -forward.
 */
function _quatFromBasis(right: vec3, up: vec3, forward: vec3): quat {
    // Rotation matrix (column-major convention, but we read row-major here):
    //   m00=right.x   m01=up.x   m02=-forward.x
    //   m10=right.y   m11=up.y   m12=-forward.y
    //   m20=right.z   m21=up.z   m22=-forward.z
    const m00 = right[0]!,   m01 = up[0]!,   m02 = -forward[0]!;
    const m10 = right[1]!,   m11 = up[1]!,   m12 = -forward[1]!;
    const m20 = right[2]!,   m21 = up[2]!,   m22 = -forward[2]!;
    const trace = m00 + m11 + m22;
    const out = quat.create();
    if (trace > 0) {
        const s = 0.5 / Math.sqrt(trace + 1.0);
        out[3] = 0.25 / s;
        out[0] = (m21 - m12) * s;
        out[1] = (m02 - m20) * s;
        out[2] = (m10 - m01) * s;
    } else if (m00 > m11 && m00 > m22) {
        const s = 2.0 * Math.sqrt(1.0 + m00 - m11 - m22);
        out[3] = (m21 - m12) / s;
        out[0] = 0.25 * s;
        out[1] = (m01 + m10) / s;
        out[2] = (m02 + m20) / s;
    } else if (m11 > m22) {
        const s = 2.0 * Math.sqrt(1.0 + m11 - m00 - m22);
        out[3] = (m02 - m20) / s;
        out[0] = (m01 + m10) / s;
        out[1] = 0.25 * s;
        out[2] = (m12 + m21) / s;
    } else {
        const s = 2.0 * Math.sqrt(1.0 + m22 - m00 - m11);
        out[3] = (m10 - m01) / s;
        out[0] = (m02 + m20) / s;
        out[1] = (m12 + m21) / s;
        out[2] = 0.25 * s;
    }
    quat.normalize(out, out);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// FreeLookController  — WASD + mouse-look  (default)
// ─────────────────────────────────────────────────────────────────────────────

export interface FreeLookOptions {
    /**
     * Base translation speed in world units per second.
     * @default 5
     */
    moveSpeed?: number;
    /**
     * Mouse sensitivity in radians per pixel.
     * @default 0.002
     */
    lookSensitivity?: number;
    /**
     * Speed multiplier applied while Shift is held.
     * @default 3
     */
    sprintMultiplier?: number;
}

/**
 * Classic fly / first-person camera controller.
 *
 * Controls
 * ────────
 *   W / S            Forward / backward  (camera -Z / +Z)
 *   A / D            Strafe left / right (camera -X / +X)
 *   E  / Q           Fly up / down       (world +Y / -Y)
 *   Space            Fly up              (world +Y, alternative)
 *   Shift (L or R)   Sprint
 *   Mouse drag       Look — requires pointer lock.
 *                    Click the viewport once to acquire lock;
 *                    press Escape to release.
 */
export class FreeLookController implements CameraController {

    readonly name = 'FreeLook';
    enabled = true;
    device: InputDevice = 'mouse';
    invertX = false;
    invertY = false;

    moveSpeed        = 5.0;
    lookSensitivity  = 0.002;
    sprintMultiplier = 3.0;

    /** Gamepad look sensitivity in radians per second at full stick deflection. */
    gamepadLookSpeed = 2.0;

    private _yaw   = 0; // rotation around world Y (left / right)
    private _pitch = 0; // rotation around local X  (up / down)
    private _pos   = vec3.create();
    private _synced = false; // false → sync position from node on next update

    constructor(options: FreeLookOptions = {}) {
        if (options.moveSpeed        !== undefined) this.moveSpeed        = options.moveSpeed;
        if (options.lookSensitivity  !== undefined) this.lookSensitivity  = options.lookSensitivity;
        if (options.sprintMultiplier !== undefined) this.sprintMultiplier = options.sprintMultiplier;
    }

    onActivate(_canvas: HTMLCanvasElement): void {
        // Re-sync position from the scene node when (re-)activated so the
        // controller starts from wherever the camera currently is.
        this._synced = false;
    }

    update(dt: number, input: InputState, scene: SceneGraph, cameraNode: NodeHandle): void {
        const node = scene.getNode(cameraNode);
        if (!node) return;

        // Sync start position and rotation from the scene node on first update.
        if (!this._synced) {
            const t = node.localTransform;
            vec3.set(this._pos, t.position[0]!, t.position[1]!, t.position[2]!);
            const angles = _yawPitchFromQuat(t.rotation as Float32Array);
            this._yaw   = angles.yaw;
            this._pitch = angles.pitch;
            this._synced = true;
        }

        if (this.device === 'gamepad') {
            this._updateGamepad(dt, input, scene, cameraNode);
        } else {
            this._updateMouse(dt, input, scene, cameraNode);
        }
    }

    private _updateMouse(dt: number, input: InputState, scene: SceneGraph, cameraNode: NodeHandle): void {
        // ── Rotation (mouse) ────────────────────────────────────────────────
        if (input.mouse.locked) {
            const ix = this.invertX ? 1 : -1;
            const iy = this.invertY ? 1 : -1;
            this._yaw   += ix * input.mouse.dx * this.lookSensitivity;
            this._pitch += iy * input.mouse.dy * this.lookSensitivity;
            this._pitch  = Math.max(-HALF_PI, Math.min(HALF_PI, this._pitch));
        }

        const rotation = _quatFromYawPitch(this._yaw, this._pitch);

        // ── Translation (keyboard) ──────────────────────────────────────────
        const kb = input.keyboard;
        const sprint = kb.isDown('ShiftLeft') || kb.isDown('ShiftRight');
        const speed  = dt * this.moveSpeed * (sprint ? this.sprintMultiplier : 1);

        if (speed > 0) {
            const forward = vec3.fromValues(0, 0, -1);
            vec3.transformQuat(forward, forward, rotation);

            const right = vec3.fromValues(1, 0, 0);
            vec3.transformQuat(right, right, rotation);

            if (kb.isDown('KeyW')) vec3.scaleAndAdd(this._pos, this._pos, forward,  speed);
            if (kb.isDown('KeyS')) vec3.scaleAndAdd(this._pos, this._pos, forward, -speed);
            if (kb.isDown('KeyD')) vec3.scaleAndAdd(this._pos, this._pos, right,    speed);
            if (kb.isDown('KeyA')) vec3.scaleAndAdd(this._pos, this._pos, right,   -speed);

            if (kb.isDown('KeyE') || kb.isDown('Space'))       this._pos[1]! += speed;
            if (kb.isDown('KeyQ') || kb.isDown('ControlLeft')) this._pos[1]! -= speed;
        }

        _applyTransform(scene, cameraNode, this._pos, rotation);
    }

    private _updateGamepad(dt: number, input: InputState, scene: SceneGraph, cameraNode: NodeHandle): void {
        const gp = input.gamepad;
        if (!gp.connected) return;

        // ── Rotation (right stick) ──────────────────────────────────────────
        const rx = gp.axes[2] ?? 0;
        const ry = gp.axes[3] ?? 0;
        const ix = this.invertX ? 1 : -1;
        const iy = this.invertY ? 1 : -1;
        this._yaw   += ix * rx * this.gamepadLookSpeed * dt;
        this._pitch += iy * ry * this.gamepadLookSpeed * dt;
        this._pitch  = Math.max(-HALF_PI, Math.min(HALF_PI, this._pitch));

        const rotation = _quatFromYawPitch(this._yaw, this._pitch);

        // ── Translation (left stick) ────────────────────────────────────────
        const lx = gp.axes[0] ?? 0;
        const ly = gp.axes[1] ?? 0;
        const speed = dt * this.moveSpeed;

        const forward = vec3.fromValues(0, 0, -1);
        vec3.transformQuat(forward, forward, rotation);
        const right = vec3.fromValues(1, 0, 0);
        vec3.transformQuat(right, right, rotation);

        // Left stick Y: forward/backward (negative Y = push up = forward)
        vec3.scaleAndAdd(this._pos, this._pos, forward, -ly * speed);
        // Left stick X: strafe left/right
        vec3.scaleAndAdd(this._pos, this._pos, right, lx * speed);

        _applyTransform(scene, cameraNode, this._pos, rotation);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OrbitController  — turntable camera orbiting a fixed point
// ─────────────────────────────────────────────────────────────────────────────

export interface OrbitOptions {
    /**
     * Mouse drag sensitivity for orbiting (radians per pixel).
     * @default 0.005
     */
    orbitSensitivity?: number;
    /**
     * Scroll wheel zoom sensitivity.
     * @default 0.001
     */
    zoomSensitivity?: number;
    /**
     * Initial distance from the target point.
     * @default 5
     */
    distance?: number;
    /**
     * Minimum orbit distance (clamped on zoom in).
     * @default 0.2
     */
    minDistance?: number;
    /**
     * Maximum orbit distance (clamped on zoom out).
     * @default 100
     */
    maxDistance?: number;
}

/**
 * Turntable orbit camera — the camera orbits around the scene origin.
 * Feels like rotating the object under the cursor.
 *
 * Controls
 * ────────
 *   LMB drag         Orbit (rotate) around the target point.
 *   Scroll wheel     Zoom in / out (change orbit distance).
 *
 * No keyboard controls — WASD is intentionally disabled.
 * No pointer lock — uses regular mouse drag.
 */
export class OrbitController implements CameraController {

    readonly name = 'Orbit';
    readonly wantsPointerLock = false;
    enabled = true;
    device: InputDevice = 'mouse';
    invertX = false;
    invertY = false;

    orbitSensitivity = 0.005;
    zoomSensitivity  = 0.001;
    distance         = 5.0;
    minDistance       = 0.2;
    maxDistance       = 100.0;

    /** Gamepad orbit speed in radians per second at full stick deflection. */
    gamepadOrbitSpeed = 2.0;
    /** Gamepad zoom speed (multiplicative per second at full stick deflection). */
    gamepadZoomSpeed  = 2.0;

    // Spherical angles (azimuth around Y, elevation from XZ plane).
    private _yaw   = 0;
    private _pitch = 0.3; // slightly above the equator

    constructor(options: OrbitOptions = {}) {
        if (options.orbitSensitivity !== undefined) this.orbitSensitivity = options.orbitSensitivity;
        if (options.zoomSensitivity  !== undefined) this.zoomSensitivity  = options.zoomSensitivity;
        if (options.distance         !== undefined) this.distance         = options.distance;
        if (options.minDistance       !== undefined) this.minDistance      = options.minDistance;
        if (options.maxDistance       !== undefined) this.maxDistance      = options.maxDistance;
    }

    private _synced = false;

    onActivate(_canvas: HTMLCanvasElement): void {
        this._synced = false;
    }

    update(dt: number, input: InputState, scene: SceneGraph, cameraNode: NodeHandle): void {
        // Sync yaw/pitch/distance from the scene node on first update.
        if (!this._synced) {
            const node = scene.getNode(cameraNode);
            if (node) {
                const p = node.localTransform.position;
                const x = p[0]!, y = p[1]!, z = p[2]!;
                this.distance = Math.sqrt(x * x + y * y + z * z);
                if (this.distance > 1e-6) {
                    this._pitch = Math.asin(Math.max(-1, Math.min(1, y / this.distance)));
                    this._yaw   = Math.atan2(x, z);
                }
            }
            this._synced = true;
        }

        if (this.device === 'gamepad') {
            this._updateGamepad(dt, input);
        } else {
            this._updateMouse(input);
        }

        this._applyCameraTransform(scene, cameraNode);
    }

    private _updateMouse(input: InputState): void {
        // ── Orbit rotation (LMB drag) ───────────────────────────────────────
        if (input.mouse.isButtonDown(0)) {
            const ix = this.invertX ? -1 : 1;
            const iy = this.invertY ? -1 : 1;
            this._yaw   += ix * input.mouse.dx * this.orbitSensitivity;
            this._pitch += iy * input.mouse.dy * this.orbitSensitivity;
            this._pitch  = Math.max(-HALF_PI, Math.min(HALF_PI, this._pitch));
        }

        // ── Zoom (scroll wheel) ─────────────────────────────────────────────
        if (input.mouse.scrollDelta !== 0) {
            this.distance *= 1 + input.mouse.scrollDelta * this.zoomSensitivity;
            this.distance  = Math.max(this.minDistance, Math.min(this.maxDistance, this.distance));
        }
    }

    private _updateGamepad(dt: number, input: InputState): void {
        const gp = input.gamepad;
        if (!gp.connected) return;

        // ── Rotation (right stick) ──────────────────────────────────────────
        const rx = gp.axes[2] ?? 0;
        const ry = gp.axes[3] ?? 0;
        const ix = this.invertX ? -1 : 1;
        const iy = this.invertY ? -1 : 1;
        this._yaw   += ix * rx * this.gamepadOrbitSpeed * dt;
        this._pitch += iy * ry * this.gamepadOrbitSpeed * dt;
        this._pitch  = Math.max(-HALF_PI, Math.min(HALF_PI, this._pitch));

        // ── Zoom (left stick Y) ─────────────────────────────────────────────
        const ly = gp.axes[1] ?? 0;
        if (ly !== 0) {
            this.distance *= 1 + ly * this.gamepadZoomSpeed * dt;
            this.distance  = Math.max(this.minDistance, Math.min(this.maxDistance, this.distance));
        }
    }

    private _applyCameraTransform(scene: SceneGraph, cameraNode: NodeHandle): void {
        const cosPitch = Math.cos(this._pitch);
        const camPos = vec3.fromValues(
            this.distance * cosPitch * Math.sin(this._yaw),
            this.distance * Math.sin(this._pitch),
            this.distance * cosPitch * Math.cos(this._yaw),
        );

        const forward = vec3.create();
        vec3.negate(forward, camPos);
        vec3.normalize(forward, forward);

        const worldUp = vec3.fromValues(0, 1, 0);
        const right   = vec3.create();
        vec3.cross(right, forward, worldUp);
        vec3.normalize(right, right);

        const up = vec3.create();
        vec3.cross(up, right, forward);

        const rotation = _quatFromBasis(right, up, forward);
        _applyTransform(scene, cameraNode, camPos, rotation);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EditorController  — 3D-editor-style camera (Maya / Blender)
// ─────────────────────────────────────────────────────────────────────────────

export interface EditorOptions {
    /**
     * Mouse drag sensitivity for orbiting (radians per pixel).
     * @default 0.005
     */
    orbitSensitivity?: number;
    /**
     * Mouse drag sensitivity for panning (world units per pixel).
     * Automatically scaled by current distance to target.
     * @default 0.002
     */
    panSensitivity?: number;
    /**
     * Scroll wheel zoom sensitivity (multiplicative).
     * @default 0.001
     */
    zoomSensitivity?: number;
    /**
     * Initial distance from the target point.
     * @default 5
     */
    distance?: number;
    /**
     * Minimum orbit distance (clamped on zoom in).
     * @default 0.1
     */
    minDistance?: number;
    /**
     * Maximum orbit distance (clamped on zoom out).
     * @default 200
     */
    maxDistance?: number;
    /**
     * Initial target point the camera orbits around.
     * @default [0, 0, 0]
     */
    target?: [number, number, number];
}

/**
 * 3D editor camera — orbits a moveable target point.
 *
 * Controls
 * ────────
 *   LMB drag         Pan — translate the target in screen-local XY.
 *   RMB drag         Orbit (rotate) around the target point.
 *   Scroll wheel     Dolly — move toward / away from target.
 *
 * No pointer lock — uses regular mouse drag.
 */
export class EditorController implements CameraController {

    readonly name = 'Editor';
    readonly wantsPointerLock = false;
    enabled = true;
    device: InputDevice = 'mouse';
    invertX = false;
    invertY = false;

    orbitSensitivity = 0.005;
    panSensitivity   = 0.002;
    zoomSensitivity  = 0.001;
    distance         = 5.0;
    minDistance       = 0.1;
    maxDistance       = 200.0;

    /** Gamepad orbit speed in radians per second at full stick deflection. */
    gamepadOrbitSpeed = 2.0;
    /** Gamepad pan speed in world units per second at full stick deflection. */
    gamepadPanSpeed   = 3.0;
    /** Gamepad zoom speed (multiplicative per second at full stick deflection). */
    gamepadZoomSpeed  = 2.0;

    // Spherical angles around the target.
    private _yaw   = 0;
    private _pitch = 0.3;

    // Target point the camera orbits around (world space).
    private _target = vec3.fromValues(0, 0, 0);

    private _synced = false;

    constructor(options: EditorOptions = {}) {
        if (options.orbitSensitivity !== undefined) this.orbitSensitivity = options.orbitSensitivity;
        if (options.panSensitivity   !== undefined) this.panSensitivity   = options.panSensitivity;
        if (options.zoomSensitivity  !== undefined) this.zoomSensitivity  = options.zoomSensitivity;
        if (options.distance         !== undefined) this.distance         = options.distance;
        if (options.minDistance       !== undefined) this.minDistance      = options.minDistance;
        if (options.maxDistance       !== undefined) this.maxDistance      = options.maxDistance;
        if (options.target           !== undefined) vec3.set(this._target, options.target[0], options.target[1], options.target[2]);
    }

    onActivate(_canvas: HTMLCanvasElement): void {
        this._synced = false;
    }

    update(dt: number, input: InputState, scene: SceneGraph, cameraNode: NodeHandle): void {
        // Sync yaw/pitch/distance from the scene node on first update.
        if (!this._synced) {
            const node = scene.getNode(cameraNode);
            if (node) {
                const p = node.localTransform.position;
                // Camera position relative to target.
                const rx = p[0]! - this._target[0]!;
                const ry = p[1]! - this._target[1]!;
                const rz = p[2]! - this._target[2]!;
                this.distance = Math.sqrt(rx * rx + ry * ry + rz * rz);
                if (this.distance > 1e-6) {
                    this._pitch = Math.asin(Math.max(-1, Math.min(1, ry / this.distance)));
                    this._yaw   = Math.atan2(rx, rz);
                }
            }
            this._synced = true;
        }

        if (this.device === 'gamepad') {
            this._updateGamepad(dt, input);
        } else {
            this._updateMouse(input);
        }

        this._applyCameraTransform(scene, cameraNode);
    }

    private _updateMouse(input: InputState): void {
        // ── Orbit rotation (RMB drag) ───────────────────────────────────────
        if (input.mouse.isButtonDown(2)) {
            const ix = this.invertX ? -1 : 1;
            const iy = this.invertY ? -1 : 1;
            this._yaw   += ix * input.mouse.dx * this.orbitSensitivity;
            this._pitch += iy * input.mouse.dy * this.orbitSensitivity;
            this._pitch  = Math.max(-HALF_PI, Math.min(HALF_PI, this._pitch));
        }

        // ── Pan (LMB drag) ─────────────────────────────────────────────────
        if (input.mouse.isButtonDown(0)) {
            const panScale = this.panSensitivity * this.distance;
            const cosPitch = Math.cos(this._pitch);
            const sinYaw   = Math.sin(this._yaw);
            const cosYaw   = Math.cos(this._yaw);

            const rightX = cosYaw;
            const rightZ = -sinYaw;

            const upX = -sinYaw * -Math.sin(this._pitch);
            const upY = cosPitch;
            const upZ = -cosYaw * -Math.sin(this._pitch);

            const dx = -input.mouse.dx * panScale;
            const dy =  input.mouse.dy * panScale;

            this._target[0]! += rightX * dx + upX * dy;
            this._target[1]! +=               upY * dy;
            this._target[2]! += rightZ * dx + upZ * dy;
        }

        // ── Dolly (scroll wheel) ─────────────────────────────────────────────
        if (input.mouse.scrollDelta !== 0) {
            this.distance *= 1 + input.mouse.scrollDelta * this.zoomSensitivity;
            this.distance  = Math.max(this.minDistance, Math.min(this.maxDistance, this.distance));
        }
    }

    private _updateGamepad(dt: number, input: InputState): void {
        const gp = input.gamepad;
        if (!gp.connected) return;

        // ── Rotation (right stick) ──────────────────────────────────────────
        const rx = gp.axes[2] ?? 0;
        const ry = gp.axes[3] ?? 0;
        const ix = this.invertX ? -1 : 1;
        const iy = this.invertY ? -1 : 1;
        this._yaw   += ix * rx * this.gamepadOrbitSpeed * dt;
        this._pitch += iy * ry * this.gamepadOrbitSpeed * dt;
        this._pitch  = Math.max(-HALF_PI, Math.min(HALF_PI, this._pitch));

        // ── Pan (left stick X — horizontal translate) ───────────────────────
        const lx = gp.axes[0] ?? 0;
        if (lx !== 0) {
            const cosYaw = Math.cos(this._yaw);
            const sinYaw = Math.sin(this._yaw);
            const panAmount = lx * this.gamepadPanSpeed * dt;
            // Move target along the camera-local right direction
            this._target[0]! += cosYaw * panAmount;
            this._target[2]! += -sinYaw * panAmount;
        }

        // ── Zoom (left stick Y — dolly in/out) ─────────────────────────────
        const ly = gp.axes[1] ?? 0;
        if (ly !== 0) {
            this.distance *= 1 + ly * this.gamepadZoomSpeed * dt;
            this.distance  = Math.max(this.minDistance, Math.min(this.maxDistance, this.distance));
        }
    }

    private _applyCameraTransform(scene: SceneGraph, cameraNode: NodeHandle): void {
        const cosPitch = Math.cos(this._pitch);
        const camPos = vec3.fromValues(
            this._target[0]! + this.distance * cosPitch * Math.sin(this._yaw),
            this._target[1]! + this.distance * Math.sin(this._pitch),
            this._target[2]! + this.distance * cosPitch * Math.cos(this._yaw),
        );

        const forward = vec3.create();
        vec3.subtract(forward, this._target as vec3, camPos);
        vec3.normalize(forward, forward);

        const worldUp = vec3.fromValues(0, 1, 0);
        const right   = vec3.create();
        vec3.cross(right, forward, worldUp);
        vec3.normalize(right, right);

        const up = vec3.create();
        vec3.cross(up, right, forward);

        const rotation = _quatFromBasis(right, up, forward);
        _applyTransform(scene, cameraNode, camPos, rotation);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CameraControllerSystem
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Manages a registry of camera controllers and routes per-frame input to the
 * currently active one.
 *
 * External code can disable the whole system at any time — e.g. during a
 * cutscene or a UI menu — with `setEnabled(false)`.  The active controller
 * and its internal state (position, yaw, pitch) are preserved, so
 * re-enabling resumes seamlessly from the same pose.
 *
 * Usage
 * ─────
 *   system.init(canvas);
 *
 *   system.register(new FreeLookController());
 *
 *   system.attachCamera(camNodeHandle, scene);
 *   system.setActive('FreeLook');   // ← default
 *
 *   // each frame:
 *   system.update(dt, inputState);
 *
 *   // pause for cutscene:
 *   system.setEnabled(false);
 *   // resume:
 *   system.setEnabled(true);
 */
export class CameraControllerSystem {

    private _enabled     = true;
    private _active:     CameraController | null = null;
    private _controllers = new Map<string, CameraController>();

    /** All built-in controllers instantiated with defaults — always available. */
    private _available   = new Map<string, CameraController>();

    private _cameraNode: NodeHandle | null = null;
    private _scene:      SceneGraph | null = null;
    private _canvas:     HTMLCanvasElement | null = null;
    private _inputSystem: InputSystem | null = null;

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    init(canvas: HTMLCanvasElement, inputSystem?: InputSystem): void {
        this._canvas = canvas;
        this._inputSystem = inputSystem ?? null;

        // Populate the available pool with default instances of every built-in controller.
        this._available.clear();
        this._available.set('FreeLook', new FreeLookController());
        this._available.set('Orbit',    new OrbitController());
        this._available.set('Editor',   new EditorController());
    }

    destroy(): void {
        for (const ctrl of this._controllers.values()) ctrl.destroy?.();
        this._controllers.clear();
        for (const ctrl of this._available.values()) ctrl.destroy?.();
        this._available.clear();
        this._active      = null;
        this._cameraNode  = null;
        this._scene       = null;
        this._canvas      = null;
        this._inputSystem = null;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Controller registry
    // ─────────────────────────────────────────────────────────────────────────

    register(controller: CameraController): void {
        this._controllers.set(controller.name, controller);
        // Keep the available pool in sync — scene-loaded controllers with
        // custom config replace the default-constructed instance.
        this._available.set(controller.name, controller);
    }

    unregister(name: string): void {
        if (this._active?.name === name) this.setActive(null);
        const ctrl = this._controllers.get(name);
        ctrl?.destroy?.();
        this._controllers.delete(name);
    }

    /** Retrieve a registered controller by name (typed convenience overload). */
    getController<T extends CameraController>(name: string): T | undefined {
        return this._controllers.get(name) as T | undefined;
    }

    /** Return the names of all registered (scene-loaded) controllers. */
    getRegisteredNames(): string[] {
        return Array.from(this._controllers.keys());
    }

    /** Return the names of all available controllers (built-in defaults). */
    getAvailableNames(): string[] {
        return Array.from(this._available.keys());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Active controller selection
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Make the named controller active.
     * Pass `null` to have no active controller (camera frozen).
     */
    setActive(name: string | null): void {
        // Deactivate current
        if (this._active) {
            this._active.onDeactivate?.();
            this._active = null;
        }

        if (name === null) return;

        let ctrl = this._controllers.get(name);
        if (!ctrl) {
            // Auto-register from the available pool if present.
            const avail = this._available.get(name);
            if (!avail) {
                console.warn(`[CameraControllerSystem] Controller "${name}" is not registered.`);
                return;
            }
            this.register(avail);
            ctrl = avail;
        }
        this._active = ctrl;
        if (this._inputSystem) {
            // Only request pointer lock for mouse-driven controllers that want it.
            const wantsLock = ctrl.device === 'mouse' && ctrl.wantsPointerLock !== false;
            this._inputSystem.pointerLockEnabled = wantsLock;
            if (!wantsLock && document.pointerLockElement) {
                document.exitPointerLock();
            }
        }
        if (this._canvas) ctrl.onActivate?.(this._canvas);
    }

    getActive(): CameraController | null { return this._active; }

    /** Set the input device for a registered/available controller by name. */
    setDevice(name: string, device: InputDevice): void {
        const ctrl = this._controllers.get(name) ?? this._available.get(name);
        if (!ctrl) return;
        ctrl.device = device;
        // If this is the active controller, update pointer lock state.
        if (ctrl === this._active && this._inputSystem) {
            const wantsLock = device === 'mouse' && ctrl.wantsPointerLock !== false;
            this._inputSystem.pointerLockEnabled = wantsLock;
            if (!wantsLock && document.pointerLockElement) {
                document.exitPointerLock();
            }
        }
    }

    /** Get the current input device for a controller by name. */
    getDevice(name: string): InputDevice | undefined {
        const ctrl = this._controllers.get(name) ?? this._available.get(name);
        return ctrl?.device;
    }

    /** Set horizontal axis inversion for a controller by name. */
    setInvertX(name: string, invert: boolean): void {
        const ctrl = this._controllers.get(name) ?? this._available.get(name);
        if (ctrl) ctrl.invertX = invert;
    }

    /** Get horizontal axis inversion for a controller by name. */
    getInvertX(name: string): boolean {
        const ctrl = this._controllers.get(name) ?? this._available.get(name);
        return ctrl?.invertX ?? false;
    }

    /** Set vertical axis inversion for a controller by name. */
    setInvertY(name: string, invert: boolean): void {
        const ctrl = this._controllers.get(name) ?? this._available.get(name);
        if (ctrl) ctrl.invertY = invert;
    }

    /** Get vertical axis inversion for a controller by name. */
    getInvertY(name: string): boolean {
        const ctrl = this._controllers.get(name) ?? this._available.get(name);
        return ctrl?.invertY ?? false;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Camera binding
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Bind the controller system to a camera scene node.
     * All registered controllers will move this node when active.
     */
    attachCamera(nodeHandle: NodeHandle, scene: SceneGraph): void {
        this._cameraNode = nodeHandle;
        this._scene      = scene;
    }

    detachCamera(): void {
        this._cameraNode = null;
        this._scene      = null;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Enable / disable  (cutscene / UI takeover)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Enable or disable the entire controller system.
     *
     * Disabling freezes the camera in place without losing the active
     * controller's state.  Re-enabling resumes from the same pose.
     */
    setEnabled(enabled: boolean): void { this._enabled = enabled; }
    isEnabled(): boolean               { return this._enabled; }

    // ─────────────────────────────────────────────────────────────────────────
    // Per-frame update  (called by FrameOrchestrator)
    // ─────────────────────────────────────────────────────────────────────────

    update(dt: number, input: InputState): void {
        if (!this._enabled)           return;
        if (!this._active?.enabled)   return;
        if (this._cameraNode === null) return;
        if (!this._scene)             return;

        this._active.update(dt, input, this._scene, this._cameraNode);
    }
}
