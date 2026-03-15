// /src/engine/scene/BVH.ts
//
// Two-tier acceleration structure for frustum culling:
//   1. Static BVH — SAH-built binary tree over world-space AABBs of static mesh nodes.
//   2. Dynamic flat list — linearly scanned each frame (handled by the caller).
//
// Also used for per-cascade shadow culling so the shadow pass only draws
// geometry that intersects a given cascade's frustum.

import type { AABB, Vec3, MeshHandle } from '../geometry/MeshSystem';
import type { Mat4, NodeHandle } from './SceneGraph';
import type { MaterialHandle } from '../materials/MaterialSystem';

// -------------------------------------------------------------------------
// World-space AABB from object-space AABB + world matrix
// -------------------------------------------------------------------------

/**
 * Compute a tight axis-aligned bounding box of an object-space AABB
 * transformed by a 4×4 column-major world matrix.
 *
 * Uses the absolute-value decomposition method (Arvo 1990) which is
 * faster and tighter than transforming all 8 corners.
 */
export function computeWorldAABB(local: AABB, m: Mat4): AABB {
    const localCx = (local.min.x + local.max.x) * 0.5;
    const localCy = (local.min.y + local.max.y) * 0.5;
    const localCz = (local.min.z + local.max.z) * 0.5;
    const halfX   = (local.max.x - local.min.x) * 0.5;
    const halfY   = (local.max.y - local.min.y) * 0.5;
    const halfZ   = (local.max.z - local.min.z) * 0.5;

    // Transform center
    const cx = m[0]! * localCx + m[4]! * localCy + m[8]!  * localCz + m[12]!;
    const cy = m[1]! * localCx + m[5]! * localCy + m[9]!  * localCz + m[13]!;
    const cz = m[2]! * localCx + m[6]! * localCy + m[10]! * localCz + m[14]!;

    // Compute world-space half-extents using absolute matrix columns
    const ex = Math.abs(m[0]!) * halfX + Math.abs(m[4]!) * halfY + Math.abs(m[8]!)  * halfZ;
    const ey = Math.abs(m[1]!) * halfX + Math.abs(m[5]!) * halfY + Math.abs(m[9]!)  * halfZ;
    const ez = Math.abs(m[2]!) * halfX + Math.abs(m[6]!) * halfY + Math.abs(m[10]!) * halfZ;

    return {
        min: { x: cx - ex, y: cy - ey, z: cz - ez },
        max: { x: cx + ex, y: cy + ey, z: cz + ez },
    };
}

// -------------------------------------------------------------------------
// AABB-vs-frustum test (p-vertex / n-vertex)
// -------------------------------------------------------------------------

/**
 * Test whether an AABB is at least partially inside a frustum defined by
 * 6 normalised planes [a, b, c, d] each (Float32Array of length 4).
 *
 * Uses the p-vertex technique: for each plane, pick the AABB corner
 * furthest along the plane normal. If even that corner is behind the
 * plane, the entire box is outside.
 */
export function aabbInFrustum(aabb: AABB, planes: Float32Array[]): boolean {
    const minX = aabb.min.x, minY = aabb.min.y, minZ = aabb.min.z;
    const maxX = aabb.max.x, maxY = aabb.max.y, maxZ = aabb.max.z;

    for (let i = 0; i < planes.length; i++) {
        const p = planes[i]!;
        const a = p[0]!, b = p[1]!, c = p[2]!, d = p[3]!;

        // p-vertex: the corner with the maximum signed distance from the plane
        const px = a >= 0 ? maxX : minX;
        const py = b >= 0 ? maxY : minY;
        const pz = c >= 0 ? maxZ : minZ;

        if (a * px + b * py + c * pz + d < 0) return false;
    }
    return true;
}

// -------------------------------------------------------------------------
// BVH leaf entry
// -------------------------------------------------------------------------

export interface BVHLeafEntry {
    nodeHandle:      NodeHandle;
    meshHandle:      MeshHandle;
    materialHandles: MaterialHandle[];
    worldMatrix:     Mat4;
    worldAABB:       AABB;
    isStatic:        boolean;
}

// -------------------------------------------------------------------------
// BVH node (flat array-of-structs layout)
// -------------------------------------------------------------------------

interface BVHNode {
    aabb:       AABB;
    leftChild:  number;   // index into nodes[] (-1 for leaf)
    rightChild: number;   // index into nodes[] (-1 for leaf)
    /** Index range [leafStart, leafEnd) into the leaves array. Only valid for leaf nodes. */
    leafStart:  number;
    leafEnd:    number;
}

// -------------------------------------------------------------------------
// SAH BVH builder
// -------------------------------------------------------------------------

const SAH_BUCKETS = 12;
const SAH_TRAVERSAL_COST = 1.0;
const SAH_INTERSECT_COST = 1.0;
const MAX_LEAF_SIZE = 4;

function surfaceArea(aabb: AABB): number {
    const dx = aabb.max.x - aabb.min.x;
    const dy = aabb.max.y - aabb.min.y;
    const dz = aabb.max.z - aabb.min.z;
    return 2 * (dx * dy + dy * dz + dz * dx);
}

function aabbUnion(a: AABB, b: AABB): AABB {
    return {
        min: {
            x: Math.min(a.min.x, b.min.x),
            y: Math.min(a.min.y, b.min.y),
            z: Math.min(a.min.z, b.min.z),
        },
        max: {
            x: Math.max(a.max.x, b.max.x),
            y: Math.max(a.max.y, b.max.y),
            z: Math.max(a.max.z, b.max.z),
        },
    };
}

function aabbEmpty(): AABB {
    return {
        min: { x: Infinity, y: Infinity, z: Infinity },
        max: { x: -Infinity, y: -Infinity, z: -Infinity },
    };
}

function aabbExpand(aabb: AABB, point: Vec3): void {
    if (point.x < aabb.min.x) aabb.min.x = point.x;
    if (point.y < aabb.min.y) aabb.min.y = point.y;
    if (point.z < aabb.min.z) aabb.min.z = point.z;
    if (point.x > aabb.max.x) aabb.max.x = point.x;
    if (point.y > aabb.max.y) aabb.max.y = point.y;
    if (point.z > aabb.max.z) aabb.max.z = point.z;
}

function centroidAxis(aabb: AABB, axis: number): number {
    switch (axis) {
        case 0: return (aabb.min.x + aabb.max.x) * 0.5;
        case 1: return (aabb.min.y + aabb.max.y) * 0.5;
        default: return (aabb.min.z + aabb.max.z) * 0.5;
    }
}

function axisMin(aabb: AABB, axis: number): number {
    switch (axis) {
        case 0: return aabb.min.x;
        case 1: return aabb.min.y;
        default: return aabb.min.z;
    }
}

function axisMax(aabb: AABB, axis: number): number {
    switch (axis) {
        case 0: return aabb.max.x;
        case 1: return aabb.max.y;
        default: return aabb.max.z;
    }
}

// -------------------------------------------------------------------------
// StaticBVH class
// -------------------------------------------------------------------------

export class StaticBVH {
    private _nodes:  BVHNode[]       = [];
    private _leaves: BVHLeafEntry[]  = [];

    get leafCount(): number { return this._leaves.length; }

    /**
     * Build the BVH from an array of leaf entries.
     * The entries array is reordered in-place during build.
     */
    build(entries: BVHLeafEntry[]): void {
        this._leaves = entries;
        this._nodes  = [];
        if (entries.length === 0) return;
        this._buildRecursive(0, entries.length);
    }

    /**
     * Query all leaves that intersect the given frustum planes.
     * Results are pushed into the provided output array (not cleared).
     */
    query(planes: Float32Array[], out: BVHLeafEntry[]): void {
        if (this._nodes.length === 0) return;
        this._queryIterative(planes, out);
    }

    // ----- Build -----

    private _buildRecursive(start: number, end: number): number {
        const count = end - start;
        const leaves = this._leaves;

        // Compute bounds of all entries in [start, end)
        let bounds = aabbEmpty();
        for (let i = start; i < end; i++) {
            bounds = aabbUnion(bounds, leaves[i]!.worldAABB);
        }

        // Leaf node
        if (count <= MAX_LEAF_SIZE) {
            const idx = this._nodes.length;
            this._nodes.push({
                aabb: bounds,
                leftChild: -1, rightChild: -1,
                leafStart: start, leafEnd: end,
            });
            return idx;
        }

        // Find best split using SAH with bucket evaluation
        let bestCost  = Infinity;
        let bestAxis  = 0;
        let bestSplit = start + 1;

        const parentSA = surfaceArea(bounds);
        if (parentSA < 1e-10) {
            // Degenerate — make a leaf
            const idx = this._nodes.length;
            this._nodes.push({
                aabb: bounds,
                leftChild: -1, rightChild: -1,
                leafStart: start, leafEnd: end,
            });
            return idx;
        }

        for (let axis = 0; axis < 3; axis++) {
            // Compute centroid bounds along this axis
            let cMin = Infinity, cMax = -Infinity;
            for (let i = start; i < end; i++) {
                const c = centroidAxis(leaves[i]!.worldAABB, axis);
                if (c < cMin) cMin = c;
                if (c > cMax) cMax = c;
            }
            if (cMax - cMin < 1e-10) continue; // all centroids coincide on this axis

            // Assign entries to buckets
            const bucketBounds: AABB[] = [];
            const bucketCounts: number[] = [];
            for (let b = 0; b < SAH_BUCKETS; b++) {
                bucketBounds.push(aabbEmpty());
                bucketCounts.push(0);
            }

            const scale = SAH_BUCKETS / (cMax - cMin);
            for (let i = start; i < end; i++) {
                const c = centroidAxis(leaves[i]!.worldAABB, axis);
                let b = ((c - cMin) * scale) | 0;
                if (b >= SAH_BUCKETS) b = SAH_BUCKETS - 1;
                bucketCounts[b]++;
                const bb = bucketBounds[b]!;
                const la = leaves[i]!.worldAABB;
                if (la.min.x < bb.min.x) bb.min.x = la.min.x;
                if (la.min.y < bb.min.y) bb.min.y = la.min.y;
                if (la.min.z < bb.min.z) bb.min.z = la.min.z;
                if (la.max.x > bb.max.x) bb.max.x = la.max.x;
                if (la.max.y > bb.max.y) bb.max.y = la.max.y;
                if (la.max.z > bb.max.z) bb.max.z = la.max.z;
            }

            // Evaluate SAH cost for each of the SAH_BUCKETS-1 split positions
            // Sweep from left and precompute right using suffix.
            const leftBounds:  AABB[]   = [];
            const leftCounts:  number[] = [];
            const rightBounds: AABB[]   = [];
            const rightCounts: number[] = [];

            let accBounds = aabbEmpty();
            let accCount  = 0;
            for (let b = 0; b < SAH_BUCKETS - 1; b++) {
                accCount += bucketCounts[b]!;
                accBounds = aabbUnion(accBounds, bucketBounds[b]!);
                leftBounds.push({ ...accBounds, min: { ...accBounds.min }, max: { ...accBounds.max } });
                leftCounts.push(accCount);
            }

            accBounds = aabbEmpty();
            accCount  = 0;
            for (let b = SAH_BUCKETS - 1; b >= 1; b--) {
                accCount += bucketCounts[b]!;
                accBounds = aabbUnion(accBounds, bucketBounds[b]!);
                rightBounds[b - 1] = { ...accBounds, min: { ...accBounds.min }, max: { ...accBounds.max } };
                rightCounts[b - 1] = accCount;
            }

            for (let b = 0; b < SAH_BUCKETS - 1; b++) {
                const lc = leftCounts[b]!;
                const rc = rightCounts[b]!;
                if (lc === 0 || rc === 0) continue;
                const cost = SAH_TRAVERSAL_COST
                    + SAH_INTERSECT_COST * (surfaceArea(leftBounds[b]!) * lc + surfaceArea(rightBounds[b]!) * rc) / parentSA;
                if (cost < bestCost) {
                    bestCost  = cost;
                    bestAxis  = axis;
                    bestSplit = b;
                }
            }
        }

        // Partition entries by the best split
        const leafCost = SAH_INTERSECT_COST * count;
        if (bestCost >= leafCost && count <= MAX_LEAF_SIZE * 2) {
            // Cheaper to make a leaf
            const idx = this._nodes.length;
            this._nodes.push({
                aabb: bounds,
                leftChild: -1, rightChild: -1,
                leafStart: start, leafEnd: end,
            });
            return idx;
        }

        // Re-partition entries in [start, end) around the best bucket split
        {
            let cMin = Infinity, cMax = -Infinity;
            for (let i = start; i < end; i++) {
                const c = centroidAxis(leaves[i]!.worldAABB, bestAxis);
                if (c < cMin) cMin = c;
                if (c > cMax) cMax = c;
            }
            const scale = (cMax - cMin) > 1e-10 ? SAH_BUCKETS / (cMax - cMin) : 0;
            let mid = start;
            for (let i = start; i < end; i++) {
                const c = centroidAxis(leaves[i]!.worldAABB, bestAxis);
                let b = ((c - cMin) * scale) | 0;
                if (b >= SAH_BUCKETS) b = SAH_BUCKETS - 1;
                if (b <= bestSplit) {
                    // Swap to left partition
                    const tmp = leaves[mid]!;
                    leaves[mid] = leaves[i]!;
                    leaves[i] = tmp;
                    mid++;
                }
            }
            // Ensure at least one element on each side
            if (mid === start) mid = start + 1;
            if (mid === end)   mid = end - 1;

            // Allocate this internal node (index reserved before recursing)
            const idx = this._nodes.length;
            this._nodes.push({
                aabb: bounds,
                leftChild: -1, rightChild: -1,
                leafStart: -1, leafEnd: -1,
            });

            const left  = this._buildRecursive(start, mid);
            const right = this._buildRecursive(mid, end);

            this._nodes[idx]!.leftChild  = left;
            this._nodes[idx]!.rightChild = right;

            return idx;
        }
    }

    // ----- Query -----

    private _queryIterative(planes: Float32Array[], out: BVHLeafEntry[]): void {
        // Explicit stack to avoid recursive calls
        const stack: number[] = [0]; // start at root
        const nodes  = this._nodes;
        const leaves = this._leaves;

        while (stack.length > 0) {
            const idx = stack.pop()!;
            const node = nodes[idx]!;

            // Frustum test on this node's AABB
            if (!aabbInFrustum(node.aabb, planes)) continue;

            if (node.leftChild === -1) {
                // Leaf — emit all entries
                for (let i = node.leafStart; i < node.leafEnd; i++) {
                    out.push(leaves[i]!);
                }
            } else {
                // Internal — push children
                stack.push(node.rightChild);
                stack.push(node.leftChild);
            }
        }
    }
}
