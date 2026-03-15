// /src/engine/scene/SceneGraph.ts

import { mat4 } from 'gl-matrix';
import type { MeshHandle, AABB, BoundingSphere } from '../geometry/MeshSystem';
import type { MaterialHandle } from '../materials/MaterialSystem';
import type { LightHandle } from '../lights/LightSystem';

// -------------------------------------------------------------------------
// Math types (minimal; replace with a proper math library)
// -------------------------------------------------------------------------

export type Mat4 = Float32Array; // 16 floats, column-major
export type Quat = Float32Array; // 4 floats [x, y, z, w]
export type Vec3f = Float32Array; // 3 floats

export interface Transform {
    position: Vec3f;
    rotation: Quat;
    scale: Vec3f;
}

// -------------------------------------------------------------------------
// Node types
// -------------------------------------------------------------------------

export type NodeHandle = number;

export enum NodeType {
    Empty    = 'EMPTY',    // pure transform / grouping node
    Mesh     = 'MESH',     // renderable geometry
    Light    = 'LIGHT',    // light source
    Camera   = 'CAMERA',   // viewpoint
}

/**
 * A node in the scene hierarchy.
 */
export interface SceneNode {
    handle: NodeHandle;
    label: string;
    type: NodeType;
    parent: NodeHandle | null;
    children: NodeHandle[];

    /** Local-space transform relative to parent. */
    localTransform: Transform;
    /** Cached world-space matrix (recomputed when dirty). */
    worldMatrix: Mat4;
    dirty: boolean;

    /**
     * When true the node's world matrix never changes after scene load,
     * making it eligible for GPU instancing.
     *
     * Set to false for anything that moves, scales, or deforms each frame
     * (animated characters, physics objects, particles, etc.).
     *
     * TODO (Animations): skeletal, morph-target, and keyframe animations
     * must set isStatic = false so their nodes stay in the per-draw path.
     */
    isStatic: boolean;

    /**
     * When set, this node is a child instance of an instance group.
     * The editor uses this to restrict the property panel to transform-only.
     */
    isInstance?: boolean;

    // --- optional payload depending on NodeType ---
    meshHandle?: MeshHandle;
    materialHandles?: MaterialHandle[]; // one per sub-mesh
    lightHandle?: LightHandle;
    // cameraHandle omitted – Camera is its own system referencing a node
}

/**
 * Result of frustum culling: lists of visible objects grouped by render path.
 */
export interface CullResults {
    opaqueDrawables: DrawableRef[];
    transparentDrawables: DrawableRef[];
    visibleLights: NodeHandle[];
}

export interface DrawableRef {
    nodeHandle: NodeHandle;
    meshHandle: MeshHandle;
    materialHandle: MaterialHandle;
    worldMatrix: Mat4;
    /** Distance from camera (for transparency sorting). */
    distanceToCamera: number;
    /** Mirrors SceneNode.isStatic — used by FrameOrchestrator to route to instanced vs per-draw path. */
    isStatic: boolean;
}

/**
 * Manages the scene hierarchy (transform tree), world-matrix propagation,
 * frustum culling, and exposes data the render graph needs to build draw lists.
 *
 * Future RT: build the TLAS (top-level acceleration structure) from the
 * set of visible meshes + their world transforms each frame.
 */
export class SceneGraph {

    private _nextHandle: NodeHandle = 1;
    private _nodes: Map<NodeHandle, SceneNode> = new Map();
    private _rootNodes: NodeHandle[] = [];

    /**
     * Set of nodes whose subtrees need world-matrix recomputation.
     * A "dirty root" is the highest ancestor in a dirty chain — its parent
     * (if any) is NOT dirty, but the node itself and all its descendants are.
     * Maintained by _markDirty(); consumed and cleared by updateWorldMatrices().
     */
    private _dirtyRoots: Set<NodeHandle> = new Set();

    // Pooled cull result arrays — cleared and reused each frame instead of reallocated.
    private _cullOpaque:      DrawableRef[] = [];
    private _cullTransparent: DrawableRef[] = [];
    private _cullLights:      NodeHandle[]  = [];

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    init(): void {
        // no GPU dependency – purely CPU-side
    }

    // -------------------------------------------------------------------------
    // Node CRUD
    // -------------------------------------------------------------------------

    createNode(label: string, type: NodeType, parent?: NodeHandle): NodeHandle {
        const handle = this._nextHandle++;

        const node: SceneNode = {
            handle,
            label,
            type,
            parent: parent ?? null,
            children: [],
            localTransform: SceneGraph.identityTransform(),
            worldMatrix: SceneGraph.identityMat4(),
            dirty: true,
            isStatic: true,
        };

        this._nodes.set(handle, node);

        if (parent !== undefined) {
            const parentNode = this._nodes.get(parent);
            parentNode?.children.push(handle);
        } else {
            this._rootNodes.push(handle);
        }

        // New nodes need their world matrix computed on the next update.
        // If the parent is already dirty, its subtree traversal covers this
        // child — no need for a separate dirty root entry.
        const parentDirty = parent !== undefined && this._nodes.get(parent)?.dirty;
        if (!parentDirty) {
            this._dirtyRoots.add(handle);
        }

        return handle;
    }

    getNode(handle: NodeHandle): SceneNode | undefined {
        return this._nodes.get(handle);
    }

    /** Handles of all root-level nodes (no parent). */
    getRootNodes(): readonly NodeHandle[] {
        return this._rootNodes;
    }

    /** Iterate every node in the scene graph. */
    getNodes(): IterableIterator<SceneNode> {
        return this._nodes.values();
    }

    /** Find the first node whose label matches the given name, or undefined. */
    findNode(name: string): SceneNode | undefined {
        for (const node of this._nodes.values()) {
            if (node.label === name) return node;
        }
        return undefined;
    }

    destroyNode(handle: NodeHandle): void {
        // TODO: remove from parent's children, reparent children, delete
    }

    /** Returns the number of Mesh-type nodes currently in the scene. */
    getMeshNodeCount(): number {
        let count = 0;
        for (const node of this._nodes.values()) {
            if (node.type === NodeType.Mesh) count++;
        }
        return count;
    }

    /**
     * Mark a node as static or dynamic.
     *
     * Static nodes are batched into instanced GPU draw calls; dynamic nodes
     * use the per-draw uniform path.  Call this before the first rendered frame.
     * Changing isStatic at runtime is supported but triggers a batch rebuild
     * on the next frame.
     */
    setNodeStatic(handle: NodeHandle, isStatic: boolean): void {
        const node = this._nodes.get(handle);
        if (node) node.isStatic = isStatic;
    }

    /**
     * Attach a mesh + materials to an existing node.
     */
    setMeshComponent(nodeHandle: NodeHandle, meshHandle: MeshHandle, materialHandles: MaterialHandle[]): void {
        const node = this._nodes.get(nodeHandle);
        if (!node) return;
        node.type = NodeType.Mesh;
        node.meshHandle = meshHandle;
        node.materialHandles = materialHandles;
    }

    /**
     * Attach a light to an existing node.
     */
    setLightComponent(nodeHandle: NodeHandle, lightHandle: LightHandle): void {
        const node = this._nodes.get(nodeHandle);
        if (!node) return;
        node.type = NodeType.Light;
        node.lightHandle = lightHandle;
    }

    // -------------------------------------------------------------------------
    // Transform
    // -------------------------------------------------------------------------

    getLocalTransform(handle: NodeHandle): Transform | undefined {
        return this._nodes.get(handle)?.localTransform;
    }

    getWorldMatrix(handle: NodeHandle): Mat4 | undefined {
        return this._nodes.get(handle)?.worldMatrix;
    }

    setLocalTransform(handle: NodeHandle, t: Partial<Transform>): void {
        const node = this._nodes.get(handle);
        if (!node) return;
        if (t.position) node.localTransform.position = t.position;
        if (t.rotation) node.localTransform.rotation = t.rotation;
        if (t.scale)    node.localTransform.scale    = t.scale;
        this._markDirty(handle);
    }

    /**
     * Recompute world matrices for dirty subtrees only.
     *
     * Instead of traversing the full hierarchy every frame, only the subtrees
     * rooted at nodes that were dirtied since the last call are visited.
     * For mostly-static scenes this skips the vast majority of nodes.
     */
    updateWorldMatrices(): void {
        if (this._dirtyRoots.size === 0) return;

        for (const handle of this._dirtyRoots) {
            // Resolve the parent's world matrix to propagate from.
            const node = this._nodes.get(handle);
            if (!node) continue;
            let parentWorld: Mat4;
            if (node.parent !== null) {
                const parentNode = this._nodes.get(node.parent);
                parentWorld = parentNode ? parentNode.worldMatrix : SceneGraph.identityMat4();
            } else {
                parentWorld = SceneGraph.identityMat4();
            }
            this._updateRecursive(handle, parentWorld);
        }

        this._dirtyRoots.clear();
    }

    // -------------------------------------------------------------------------
    // Culling
    // -------------------------------------------------------------------------

    /**
     * Walk all nodes, frustum-cull with bounding spheres, and produce draw lists.
     *
     * @param _viewProjection     VP matrix (kept for API symmetry; frustum planes
     *                            are passed pre-extracted to avoid recomputation).
     * @param cameraPosition      World-space camera position (for distance sort).
     * @param frustumPlanes       Six normalised frustum planes [a,b,c,d] each.
     *                            Pass an empty array to skip frustum culling.
     * @param getBoundingSphere   Returns the object-space bounding sphere for a
     *                            mesh handle, or undefined if not available.
     *                            Meshes without a sphere bypass the sphere test.
     *
     * For multi-material meshes one DrawableRef is emitted per materialHandle
     * entry. Meshes with no materialHandles get a sentinel handle 0.
     *
     * @param isTransparentMaterial  Optional callback that returns true for
     *                               materials that require the forward (blend)
     *                               path.  When null all drawables are opaque.
     */
    cull(
        _viewProjection:   Mat4,
        cameraPosition:    Vec3f,
        frustumPlanes:     Float32Array[]                                       = [],
        getBoundingSphere: ((h: MeshHandle) => BoundingSphere | undefined) | null = null,
        tolerance:         number                                                = 0,
        isTransparentMaterial: ((h: MaterialHandle) => boolean) | null           = null,
    ): CullResults {
        const opaqueDrawables      = this._cullOpaque;      opaqueDrawables.length = 0;
        const transparentDrawables = this._cullTransparent;  transparentDrawables.length = 0;
        const visibleLights        = this._cullLights;       visibleLights.length = 0;

        const doCull = frustumPlanes.length === 6 && getBoundingSphere !== null;

        for (const node of this._nodes.values()) {
            // ---- Light nodes -----------------------------------------------
            if (node.type === NodeType.Light) {
                visibleLights.push(node.handle);
                continue;
            }

            // ---- Mesh nodes ------------------------------------------------
            if (node.type !== NodeType.Mesh || node.meshHandle === undefined) {
                continue;
            }

            // ---- Frustum test (sphere in world space) ----------------------
            if (doCull) {
                const sphere = getBoundingSphere!(node.meshHandle);
                if (sphere) {
                    // Transform sphere center from object space to world space.
                    // Column-major: result = M * [cx, cy, cz, 1]
                    const m  = node.worldMatrix;
                    const cx = sphere.center.x;
                    const cy = sphere.center.y;
                    const cz = sphere.center.z;
                    const wx = m[0]! * cx + m[4]! * cy + m[8]!  * cz + m[12]!;
                    const wy = m[1]! * cx + m[5]! * cy + m[9]!  * cz + m[13]!;
                    const wz = m[2]! * cx + m[6]! * cy + m[10]! * cz + m[14]!;

                    // Conservative radius: multiply by the largest column scale.
                    // Use squared scales and take sqrt only once (the max).
                    const sx2 = m[0]! * m[0]! + m[1]! * m[1]! + m[2]!  * m[2]!;
                    const sy2 = m[4]! * m[4]! + m[5]! * m[5]! + m[6]!  * m[6]!;
                    const sz2 = m[8]! * m[8]! + m[9]! * m[9]! + m[10]! * m[10]!;
                    const maxScale = Math.sqrt(Math.max(sx2, sy2, sz2));
                    const wr = sphere.radius * maxScale + tolerance;

                    if (!_sphereInFrustum(wx, wy, wz, wr, frustumPlanes)) continue;
                }
            }

            // Distance from camera for transparent depth sorting.
            // Translation lives at column 3 of the column-major world matrix.
            const tx = node.worldMatrix[12]!;
            const ty = node.worldMatrix[13]!;
            const tz = node.worldMatrix[14]!;
            const dx = tx - cameraPosition[0]!;
            const dy = ty - cameraPosition[1]!;
            const dz = tz - cameraPosition[2]!;
            const distanceToCamera = dx * dx + dy * dy + dz * dz; // squared — avoids sqrt; sort order preserved

            const handles = node.materialHandles;

            if (!handles || handles.length === 0) {
                opaqueDrawables.push({
                    nodeHandle: node.handle, meshHandle: node.meshHandle,
                    materialHandle: 0, worldMatrix: node.worldMatrix, distanceToCamera,
                    isStatic: node.isStatic,
                });
            } else {
                for (const materialHandle of handles) {
                    const ref: DrawableRef = {
                        nodeHandle: node.handle, meshHandle: node.meshHandle,
                        materialHandle, worldMatrix: node.worldMatrix, distanceToCamera,
                        isStatic: node.isStatic,
                    };
                    if (isTransparentMaterial && isTransparentMaterial(materialHandle)) {
                        transparentDrawables.push(ref);
                    } else {
                        opaqueDrawables.push(ref);
                    }
                }
            }
        }

        // Sort transparent drawables back-to-front (painter's algorithm).
        transparentDrawables.sort((a, b) => b.distanceToCamera - a.distanceToCamera);

        return { opaqueDrawables, transparentDrawables, visibleLights };
    }

    // -------------------------------------------------------------------------
    // Future: TLAS construction (ray tracing)
    // -------------------------------------------------------------------------

    /**
     * Build a TLAS from the current set of visible mesh instances.
     */
    buildTLAS(_visibleNodes: NodeHandle[]): void {
        // TODO: collect BLAS handles + world transforms, build TLAS
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * Internal recursive helper — marks node + descendants dirty without
     * touching _dirtyRoots (that's handled by the public entry point).
     */
    private _markDirtyRecursive(handle: NodeHandle): void {
        const node = this._nodes.get(handle);
        if (!node || node.dirty) return;
        node.dirty = true;
        for (const child of node.children) {
            this._markDirtyRecursive(child);
        }
    }

    /**
     * Mark a node and all its descendants as needing world-matrix recomputation.
     * The dirtied node is registered as a "dirty root" so updateWorldMatrices()
     * only traverses affected subtrees instead of the full hierarchy.
     */
    private _markDirty(handle: NodeHandle): void {
        const node = this._nodes.get(handle);
        if (!node || node.dirty) return;

        // Register as dirty root. If an ancestor is already a dirty root,
        // this node's subtree is already covered — but since ancestors are
        // NOT marked dirty by setLocalTransform (only descendants are),
        // there's no overlap: each _markDirty call creates a new dirty root.
        this._dirtyRoots.add(handle);

        // Remove any children that were previously dirty roots on their own —
        // they're now covered by this higher-level dirty root.
        for (const child of node.children) {
            this._dirtyRoots.delete(child);
        }

        node.dirty = true;
        for (const child of node.children) {
            this._markDirtyRecursive(child);
        }
    }

    private _updateRecursive(handle: NodeHandle, parentWorld: Mat4): void {
        const node = this._nodes.get(handle);
        if (!node) return;

        if (!node.dirty) {
            // _markDirty() propagates to all descendants, so if this node
            // is clean, the entire subtree is clean — skip it entirely.
            return;
        }

        const { position, rotation, scale } = node.localTransform;
        // Build the local TRS matrix, then concatenate with the parent world matrix.
        // mat4.fromRotationTranslationScale applies scale → rotate → translate.
        mat4.fromRotationTranslationScale(node.worldMatrix, rotation, position, scale);
        mat4.multiply(node.worldMatrix, parentWorld, node.worldMatrix);
        node.dirty = false;

        for (const child of node.children) {
            this._updateRecursive(child, node.worldMatrix);
        }
    }

    // -------------------------------------------------------------------------
    // Static math utilities (stubs)
    // -------------------------------------------------------------------------

    static identityTransform(): Transform {
        return {
            position: new Float32Array([0, 0, 0]),
            rotation: new Float32Array([0, 0, 0, 1]),
            scale:    new Float32Array([1, 1, 1]),
        };
    }

    static identityMat4(): Mat4 {
        const m = new Float32Array(16);
        m[0] = m[5] = m[10] = m[15] = 1;
        return m;
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    destroy(): void {
        this._nodes.clear();
        this._rootNodes.length = 0;
        this._dirtyRoots.clear();
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

/**
 * Return true if a world-space sphere is NOT completely outside any of the
 * six frustum planes (i.e. it is visible or intersecting the frustum).
 *
 * Each plane is a Float32Array(4) = [a, b, c, d] where the plane equation is
 *   a·x + b·y + c·z + d ≥ 0  →  point is on the inside half-space.
 * Planes must be normalised (unit-length normals) so the dot product gives the
 * true signed distance in world-space units.
 */
function _sphereInFrustum(
    cx: number, cy: number, cz: number,
    radius: number,
    planes: Float32Array[],
): boolean {
    for (const p of planes) {
        // Signed distance from sphere centre to plane.
        const dist = p[0]! * cx + p[1]! * cy + p[2]! * cz + p[3]!;
        // If the centre is further than `radius` behind the plane, fully outside.
        if (dist < -radius) return false;
    }
    return true;
}