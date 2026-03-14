// /src/engine/geometry/GeometryUtils.ts

import {
    VertexSemantic,
    type MeshDescriptor,
    type VertexLayoutDesc,
    type AABB,
} from './MeshSystem';

// -------------------------------------------------------------------------
// Shared interleaved vertex layout
//
//  offset  0 → POSITION  float32x3  (12 bytes)  @location(0)
//  offset 12 → NORMAL    float32x3  (12 bytes)  @location(1)
//  offset 24 → TANGENT   float32x4  (16 bytes)  @location(2)  ← Mikktspace, w = handedness ±1
//  offset 40 → UV0       float32x2  ( 8 bytes)  @location(3)
//  arrayStride: 48 bytes
// -------------------------------------------------------------------------

const INTERLEAVED_LAYOUT: VertexLayoutDesc = {
    arrayStride: 48,
    stepMode: 'vertex',
    attributes: [
        { semantic: VertexSemantic.Position, format: 'float32x3', offset:  0 },
        { semantic: VertexSemantic.Normal,   format: 'float32x3', offset: 12 },
        { semantic: VertexSemantic.Tangent,  format: 'float32x4', offset: 24 },
        { semantic: VertexSemantic.UV0,      format: 'float32x2', offset: 40 },
    ],
};

// -------------------------------------------------------------------------
// Option types
// -------------------------------------------------------------------------

export interface PlaneOptions {
    /** World-space width along the X axis. Default: 1. */
    width?: number;
    /** World-space depth along the Z axis. Default: 1. */
    depth?: number;
    /** Number of quad columns. Default: 1. */
    segmentsX?: number;
    /** Number of quad rows. Default: 1. */
    segmentsZ?: number;
}

export interface BoxOptions {
    /** World-space size along X. Default: 1. */
    width?: number;
    /** World-space size along Y. Default: 1. */
    height?: number;
    /** World-space size along Z. Default: 1. */
    depth?: number;
}

export interface UVSphereOptions {
    /** Radius of the sphere. Default: 0.5. */
    radius?: number;
    /** Number of longitudinal slices (columns). Default: 32. */
    widthSegments?: number;
    /** Number of latitudinal stacks (rows). Default: 16. */
    heightSegments?: number;
}

export interface IcoSphereOptions {
    /** Radius of the sphere. Default: 0.5. */
    radius?: number;
    /** Number of subdivision iterations. Default: 2. */
    subdivisions?: number;
}

// -------------------------------------------------------------------------
// GeometryUtils
// -------------------------------------------------------------------------

/**
 * Static factory for basic procedural shapes.
 *
 * Every method returns a {@link MeshDescriptor} ready to be passed to
 * `MeshSystem.createMesh()`.  The vertex data uses a single interleaved
 * buffer: POSITION (float32x3) | NORMAL (float32x3) | UV0 (float32x2).
 *
 * Usage:
 *   const desc = GeometryUtils.createPlane({ width: 10, depth: 10, segmentsX: 4, segmentsZ: 4 });
 *   const handle = engine.meshes.createMesh(desc);
 */
export class GeometryUtils {

    // -----------------------------------------------------------------------
    // Plane  (XZ plane, Y-up, centred at origin)
    // -----------------------------------------------------------------------

    static createPlane(options?: PlaneOptions): MeshDescriptor {
        const width   = options?.width      ?? 1;
        const depth   = options?.depth      ?? 1;
        const segX    = Math.max(1, Math.floor(options?.segmentsX ?? 1));
        const segZ    = Math.max(1, Math.floor(options?.segmentsZ ?? 1));

        const vertsX    = segX + 1;
        const vertsZ    = segZ + 1;
        const vertCount = vertsX * vertsZ;

        // 8 floats per vertex (pos3, normal3, uv2) — used for tangent computation
        const verts8 = new Float32Array(vertCount * 8);

        let vi = 0;
        for (let z = 0; z <= segZ; z++) {
            for (let x = 0; x <= segX; x++) {
                const u = x / segX;
                const v = z / segZ;
                verts8[vi++] = (u - 0.5) * width;
                verts8[vi++] = 0;
                verts8[vi++] = (v - 0.5) * depth;
                verts8[vi++] = 0; verts8[vi++] = 1; verts8[vi++] = 0; // normal Y-up
                verts8[vi++] = u; verts8[vi++] = v;
            }
        }

        // Two triangles per quad, CCW winding viewed from +Y
        const indexData = new Uint16Array(segX * segZ * 6);
        let ii = 0;
        for (let z = 0; z < segZ; z++) {
            for (let x = 0; x < segX; x++) {
                const tl = z * vertsX + x;
                const tr = tl + 1;
                const bl = tl + vertsX;
                const br = bl + 1;
                indexData[ii++] = tl; indexData[ii++] = bl; indexData[ii++] = tr;
                indexData[ii++] = tr; indexData[ii++] = bl; indexData[ii++] = br;
            }
        }

        const aabb: AABB = {
            min: { x: -width / 2, y: 0, z: -depth / 2 },
            max: { x:  width / 2, y: 0, z:  depth / 2 },
        };

        return {
            label: 'Plane',
            vertexLayouts: [INTERLEAVED_LAYOUT],
            vertexData: [_buildInterleavedWithTangents(verts8, indexData)],
            indexData,
            aabb,
        };
    }

    // -----------------------------------------------------------------------
    // Box  (axis-aligned, centred at origin, flat-shaded normals per face)
    // -----------------------------------------------------------------------

    static createBox(options?: BoxOptions): MeshDescriptor {
        const hw = (options?.width  ?? 1) / 2;
        const hh = (options?.height ?? 1) / 2;
        const hd = (options?.depth  ?? 1) / 2;

        // Each face: 4 corners defined CCW when viewed from outside.
        // Layout: [TL, TR, BL, BR] → index pattern [0,2,1, 1,2,3]
        const faces: { normal: [number, number, number]; corners: [number, number, number][] }[] = [
            // +Y (top)
            { normal: [ 0,  1,  0], corners: [[-hw, hh, -hd], [ hw, hh, -hd], [-hw, hh,  hd], [ hw, hh,  hd]] },
            // -Y (bottom)
            { normal: [ 0, -1,  0], corners: [[-hw,-hh,  hd], [ hw,-hh,  hd], [-hw,-hh, -hd], [ hw,-hh, -hd]] },
            // +Z (front)
            { normal: [ 0,  0,  1], corners: [[-hw, hh,  hd], [ hw, hh,  hd], [-hw,-hh,  hd], [ hw,-hh,  hd]] },
            // -Z (back)
            { normal: [ 0,  0, -1], corners: [[ hw, hh, -hd], [-hw, hh, -hd], [ hw,-hh, -hd], [-hw,-hh, -hd]] },
            // +X (right)
            { normal: [ 1,  0,  0], corners: [[ hw, hh,  hd], [ hw, hh, -hd], [ hw,-hh,  hd], [ hw,-hh, -hd]] },
            // -X (left)
            { normal: [-1,  0,  0], corners: [[-hw, hh, -hd], [-hw, hh,  hd], [-hw,-hh, -hd], [-hw,-hh,  hd]] },
        ];

        // 24 vertices (6 faces × 4), 36 indices (6 × 6)
        const verts8    = new Float32Array(24 * 8);
        const indexData = new Uint16Array(36);
        const uvCorners: [number, number][] = [[0, 0], [1, 0], [0, 1], [1, 1]];

        let vi = 0;
        let ii = 0;

        for (let f = 0; f < 6; f++) {
            const { normal, corners } = faces[f]!;
            const baseVertex = f * 4;

            for (let c = 0; c < 4; c++) {
                const [px, py, pz] = corners[c]!;
                const [u, v]       = uvCorners[c]!;
                verts8[vi++] = px; verts8[vi++] = py; verts8[vi++] = pz;
                verts8[vi++] = normal[0]; verts8[vi++] = normal[1]; verts8[vi++] = normal[2];
                verts8[vi++] = u; verts8[vi++] = v;
            }

            // Two triangles (CCW from outside): TL,BL,TR  TR,BL,BR
            indexData[ii++] = baseVertex + 0; indexData[ii++] = baseVertex + 2; indexData[ii++] = baseVertex + 1;
            indexData[ii++] = baseVertex + 1; indexData[ii++] = baseVertex + 2; indexData[ii++] = baseVertex + 3;
        }

        const aabb: AABB = {
            min: { x: -hw, y: -hh, z: -hd },
            max: { x:  hw, y:  hh, z:  hd },
        };

        return {
            label: 'Box',
            vertexLayouts: [INTERLEAVED_LAYOUT],
            vertexData: [_buildInterleavedWithTangents(verts8, indexData)],
            indexData,
            aabb,
        };
    }

    // -----------------------------------------------------------------------
    // UV Sphere  (latitude/longitude parameterisation, smooth normals)
    // -----------------------------------------------------------------------

    static createUVSphere(options?: UVSphereOptions): MeshDescriptor {
        const radius  = options?.radius         ?? 0.5;
        const slices  = Math.max(3, Math.floor(options?.widthSegments  ?? 32));
        const stacks  = Math.max(2, Math.floor(options?.heightSegments ?? 16));

        // (slices+1) * (stacks+1) vertices — duplicate seam column so UVs wrap cleanly
        const vertCount = (slices + 1) * (stacks + 1);
        const verts8    = new Float32Array(vertCount * 8);

        let vi = 0;
        for (let y = 0; y <= stacks; y++) {
            const phi    = (y / stacks) * Math.PI;
            const sinPhi = Math.sin(phi);
            const cosPhi = Math.cos(phi);

            for (let x = 0; x <= slices; x++) {
                const theta    = (x / slices) * 2 * Math.PI;
                const sinTheta = Math.sin(theta);
                const cosTheta = Math.cos(theta);
                const nx = sinPhi * cosTheta;
                const ny = cosPhi;
                const nz = sinPhi * sinTheta;
                verts8[vi++] = nx * radius; verts8[vi++] = ny * radius; verts8[vi++] = nz * radius;
                verts8[vi++] = nx; verts8[vi++] = ny; verts8[vi++] = nz;
                verts8[vi++] = x / slices; verts8[vi++] = y / stacks;
            }
        }

        const indexCount = slices * stacks * 6;
        const indexData  = new Uint32Array(indexCount);
        let ii = 0;

        for (let y = 0; y < stacks; y++) {
            for (let x = 0; x < slices; x++) {
                const tl = y * (slices + 1) + x;
                const tr = tl + 1;
                const bl = tl + (slices + 1);
                const br = bl + 1;
                if (y !== 0)          { indexData[ii++] = tl; indexData[ii++] = tr; indexData[ii++] = bl; }
                if (y !== stacks - 1) { indexData[ii++] = tr; indexData[ii++] = br; indexData[ii++] = bl; }
            }
        }

        const trimmedIndex = indexData.slice(0, ii);
        const aabb: AABB = {
            min: { x: -radius, y: -radius, z: -radius },
            max: { x:  radius, y:  radius, z:  radius },
        };

        return {
            label: 'UVSphere',
            vertexLayouts: [INTERLEAVED_LAYOUT],
            vertexData: [_buildInterleavedWithTangents(verts8, trimmedIndex)],
            indexData: trimmedIndex,
            aabb,
        };
    }

    // -----------------------------------------------------------------------
    // Ico Sphere  (icosahedron base, iterative midpoint subdivision)
    // -----------------------------------------------------------------------

    static createIcoSphere(options?: IcoSphereOptions): MeshDescriptor {
        const radius = options?.radius       ?? 0.5;
        const subs   = Math.max(0, Math.min(6, Math.floor(options?.subdivisions ?? 2)));

        // ── Build base icosahedron ──────────────────────────────────────────
        const t = (1 + Math.sqrt(5)) / 2; // golden ratio

        // 12 vertices of a unit icosahedron
        let positions: [number, number, number][] = [
            [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
            [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
            [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
        ];

        // Normalise to unit sphere
        positions = positions.map(([x, y, z]) => {
            const len = Math.sqrt(x * x + y * y + z * z);
            return [x / len, y / len, z / len];
        });

        // 20 faces of the icosahedron (CCW winding)
        let triangles: [number, number, number][] = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ];

        // ── Iterative midpoint subdivision ─────────────────────────────────
        const midpointCache = new Map<string, number>();

        const midpoint = (a: number, b: number): number => {
            const key = a < b ? `${a}_${b}` : `${b}_${a}`;
            const cached = midpointCache.get(key);
            if (cached !== undefined) return cached;

            const [ax, ay, az] = positions[a]!;
            const [bx, by, bz] = positions[b]!;
            const mx = (ax + bx) / 2;
            const my = (ay + by) / 2;
            const mz = (az + bz) / 2;
            const len = Math.sqrt(mx * mx + my * my + mz * mz);
            const idx = positions.length;
            positions.push([mx / len, my / len, mz / len]);
            midpointCache.set(key, idx);
            return idx;
        };

        for (let s = 0; s < subs; s++) {
            const next: [number, number, number][] = [];
            for (const [a, b, c] of triangles) {
                const ab = midpoint(a, b);
                const bc = midpoint(b, c);
                const ca = midpoint(c, a);
                next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
            }
            triangles = next;
        }

        // ── Build interleaved vertex buffer ─────────────────────────────────
        // For correct UV mapping each triangle gets its own 3 vertices
        // (avoids seam artefacts from shared vertices with different UVs).
        const vertCount = triangles.length * 3;
        const verts8    = new Float32Array(vertCount * 8);
        const indexData = new Uint32Array(vertCount);

        let vi = 0;
        let ii = 0;

        for (const [ai, bi, ci] of triangles) {
            for (const idx of [ai, bi, ci]) {
                const [nx, ny, nz] = positions[idx]!;
                const u = 0.5 + Math.atan2(nz, nx) / (2 * Math.PI);
                const v = 0.5 - Math.asin(ny) / Math.PI;
                verts8[vi++] = nx * radius; verts8[vi++] = ny * radius; verts8[vi++] = nz * radius;
                verts8[vi++] = nx; verts8[vi++] = ny; verts8[vi++] = nz;
                verts8[vi++] = u; verts8[vi++] = v;
                indexData[ii] = ii;
                ii++;
            }
        }

        const aabb: AABB = {
            min: { x: -radius, y: -radius, z: -radius },
            max: { x:  radius, y:  radius, z:  radius },
        };

        return {
            label: 'IcoSphere',
            vertexLayouts: [INTERLEAVED_LAYOUT],
            vertexData: [_buildInterleavedWithTangents(verts8, indexData)],
            indexData,
            aabb,
        };
    }
}

// -------------------------------------------------------------------------
// Private helpers
// -------------------------------------------------------------------------

/**
 * Compute Mikktspace-compatible per-vertex tangents from a flat 8-float-per-vertex
 * buffer (pos3, normal3, uv2) and a triangle index list.
 *
 * Algorithm:
 *   Per triangle: accumulate T and B vectors derived from UV-space edge equations.
 *   Per vertex:   Gram-Schmidt orthogonalize T against N, then store handedness
 *                 (w = ±1) as the sign of (N × T') · B.
 *
 * Returns a Float32Array of 4 floats per vertex: [tx, ty, tz, handedness].
 */
function _computeTangents(
    positions: Float32Array,
    normals:   Float32Array,
    uvs:       Float32Array,
    indices:   Uint16Array | Uint32Array,
): Float32Array {
    const vertCount = positions.length / 3;
    const tan1 = new Float32Array(vertCount * 3); // accumulated T
    const tan2 = new Float32Array(vertCount * 3); // accumulated B

    const triCount = indices.length / 3;
    for (let t = 0; t < triCount; t++) {
        const i0 = indices[t * 3 + 0]!;
        const i1 = indices[t * 3 + 1]!;
        const i2 = indices[t * 3 + 2]!;

        const p0x = positions[i0*3]!; const p0y = positions[i0*3+1]!; const p0z = positions[i0*3+2]!;
        const p1x = positions[i1*3]!; const p1y = positions[i1*3+1]!; const p1z = positions[i1*3+2]!;
        const p2x = positions[i2*3]!; const p2y = positions[i2*3+1]!; const p2z = positions[i2*3+2]!;

        const u0 = uvs[i0*2]!; const v0 = uvs[i0*2+1]!;
        const u1 = uvs[i1*2]!; const v1 = uvs[i1*2+1]!;
        const u2 = uvs[i2*2]!; const v2 = uvs[i2*2+1]!;

        const e1x = p1x-p0x; const e1y = p1y-p0y; const e1z = p1z-p0z;
        const e2x = p2x-p0x; const e2y = p2y-p0y; const e2z = p2z-p0z;
        const du1 = u1-u0; const dv1 = v1-v0;
        const du2 = u2-u0; const dv2 = v2-v0;

        const det = du1*dv2 - du2*dv1;
        const r   = Math.abs(det) > 1e-7 ? 1.0 / det : 0.0;

        const tx = (dv2*e1x - dv1*e2x) * r;
        const ty = (dv2*e1y - dv1*e2y) * r;
        const tz = (dv2*e1z - dv1*e2z) * r;
        const bx = (du1*e2x - du2*e1x) * r;
        const by = (du1*e2y - du2*e1y) * r;
        const bz = (du1*e2z - du2*e1z) * r;

        for (const i of [i0, i1, i2]) {
            tan1[i*3]+=tx; tan1[i*3+1]+=ty; tan1[i*3+2]+=tz;
            tan2[i*3]+=bx; tan2[i*3+1]+=by; tan2[i*3+2]+=bz;
        }
    }

    const out = new Float32Array(vertCount * 4);
    for (let i = 0; i < vertCount; i++) {
        const nx = normals[i*3]!; const ny = normals[i*3+1]!; const nz = normals[i*3+2]!;
        const tx = tan1[i*3]!;   const ty = tan1[i*3+1]!;   const tz = tan1[i*3+2]!;
        const bx = tan2[i*3]!;   const by = tan2[i*3+1]!;   const bz = tan2[i*3+2]!;

        // Gram-Schmidt orthogonalize T against N
        const dot = nx*tx + ny*ty + nz*tz;
        let ox = tx - dot*nx;
        let oy = ty - dot*ny;
        let oz = tz - dot*nz;
        const len = Math.sqrt(ox*ox + oy*oy + oz*oz);
        if (len > 1e-7) { ox /= len; oy /= len; oz /= len; }

        // Handedness: sign of (N × T') · B
        const cx = ny*oz - nz*oy;
        const cy = nz*ox - nx*oz;
        const cz = nx*oy - ny*ox;
        const w  = (cx*bx + cy*by + cz*bz) < 0.0 ? -1.0 : 1.0;

        out[i*4]   = ox; out[i*4+1] = oy; out[i*4+2] = oz; out[i*4+3] = w;
    }
    return out;
}

/**
 * Convert a flat 8-float-per-vertex buffer (pos3, normal3, uv2) to the engine's
 * 12-float-per-vertex format (pos3, normal3, tangent4, uv2) with Mikktspace tangents.
 */
function _buildInterleavedWithTangents(
    verts8:  Float32Array,
    indices: Uint16Array | Uint32Array,
): ArrayBuffer {
    const vertCount = verts8.length / 8;

    const positions = new Float32Array(vertCount * 3);
    const normals   = new Float32Array(vertCount * 3);
    const uvs       = new Float32Array(vertCount * 2);
    for (let i = 0; i < vertCount; i++) {
        positions[i*3]   = verts8[i*8]!;   positions[i*3+1] = verts8[i*8+1]!; positions[i*3+2] = verts8[i*8+2]!;
        normals  [i*3]   = verts8[i*8+3]!; normals  [i*3+1] = verts8[i*8+4]!; normals  [i*3+2] = verts8[i*8+5]!;
        uvs      [i*2]   = verts8[i*8+6]!; uvs      [i*2+1] = verts8[i*8+7]!;
    }

    const tangents = _computeTangents(positions, normals, uvs, indices);

    // Pack: pos3, normal3, tangent4, uv2  =  12 floats / vertex
    const out = new Float32Array(vertCount * 12);
    for (let i = 0; i < vertCount; i++) {
        out[i*12]    = verts8[i*8]!;    out[i*12+1]  = verts8[i*8+1]!; out[i*12+2]  = verts8[i*8+2]!; // pos
        out[i*12+3]  = verts8[i*8+3]!;  out[i*12+4]  = verts8[i*8+4]!; out[i*12+5]  = verts8[i*8+5]!; // normal
        out[i*12+6]  = tangents[i*4]!;  out[i*12+7]  = tangents[i*4+1]!;
        out[i*12+8]  = tangents[i*4+2]!; out[i*12+9] = tangents[i*4+3]!;                               // tangent
        out[i*12+10] = verts8[i*8+6]!; out[i*12+11] = verts8[i*8+7]!;                                  // uv
    }
    return out.buffer as ArrayBuffer;
}
