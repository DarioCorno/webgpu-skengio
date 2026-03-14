// /src/engine/scene/GLTFLoader.ts
//
// Loads a glTF/GLB file via @gltf-transform/core and populates the engine
// with static meshes, PBR materials, textures, and a scene-graph hierarchy.
//
// Usage:
//   const loader = new GLTFLoader();
//   await loader.load(engine, '/models/helmet.glb');
//
// Animation and skinning are parsed but NOT applied yet — see TODO markers.
// ──────────────────────────────────────────────────────────────────────────

import { WebIO, Node as GNode, Texture as GTexture, Material as GMaterial,
         Mesh as GMesh, Primitive as GPrimitive, Accessor as GAccessor,
         Skin as GSkin, Animation as GAnimation } from '@gltf-transform/core';
import type { Engine } from '../Engine';
import { NodeType } from './SceneGraph';
import type { NodeHandle } from './SceneGraph';
import { VertexSemantic } from '../geometry/MeshSystem';
import type { MeshHandle, MeshDescriptor, VertexLayoutDesc, AABB } from '../geometry/MeshSystem';
import type { MaterialHandle } from '../materials/MaterialSystem';
import { AlphaMode } from '../materials/MaterialSystem';
import type { ResourceHandle } from '../core/ResourceManager';
import { Logger } from '../core/Logger';

// ──────────────────────────────────────────────────────────────────────────
// Vertex layout: same 48-byte interleaved layout used by GeometryUtils
//
//   offset  0  POSITION  float32x3  (12 bytes)
//   offset 12  NORMAL    float32x3  (12 bytes)
//   offset 24  TANGENT   float32x4  (16 bytes)
//   offset 40  UV0       float32x2  ( 8 bytes)
//   stride: 48
// ──────────────────────────────────────────────────────────────────────────

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

// Second vertex buffer for skinning data (JOINTS_0 + WEIGHTS_0)
// Stride: 8 (uint16x4) + 16 (float32x4) = 24 bytes
const SKINNING_LAYOUT: VertexLayoutDesc = {
    arrayStride: 24,
    stepMode: 'vertex',
    attributes: [
        { semantic: VertexSemantic.Joints0,  format: 'uint16x4',  offset: 0 },
        { semantic: VertexSemantic.Weights0, format: 'float32x4', offset: 8 },
    ],
};

const FLOATS_PER_VERTEX = 12; // 48 / 4

// ──────────────────────────────────────────────────────────────────────────
// Skin / Animation placeholder types
// ──────────────────────────────────────────────────────────────────────────

/** Placeholder for a parsed skin — stored but not yet applied at runtime. */
export interface GLTFSkinData {
    name: string;
    jointNodeHandles: NodeHandle[];
    inverseBindMatrices: Float32Array; // N × 16 floats
}

/** Placeholder for a parsed animation clip. */
export interface GLTFAnimationData {
    name: string;
    channels: GLTFChannelData[];
    duration: number;
}

export interface GLTFChannelData {
    targetNodeHandle: NodeHandle;
    path: 'translation' | 'rotation' | 'scale' | 'weights';
    interpolation: 'LINEAR' | 'STEP' | 'CUBICSPLINE';
    inputTimes: Float32Array;
    outputValues: Float32Array;
}

// ──────────────────────────────────────────────────────────────────────────
// GLTFLoader
// ──────────────────────────────────────────────────────────────────────────

export class GLTFLoader {

    private readonly _log = new Logger('GLTFLoader');
    private readonly _io  = new WebIO();

    /** Skin data extracted during load (available after load() resolves). */
    readonly skins: GLTFSkinData[] = [];
    /** Animation data extracted during load (available after load() resolves). */
    readonly animations: GLTFAnimationData[] = [];
    /** Map from glTF Skin index → mesh node handle that uses it. */
    readonly skinNodeMap: Map<number, NodeHandle> = new Map();
    /** The root parent node handle passed to load() (if any). */
    rootNodeHandle: NodeHandle | undefined;

    /**
     * Load a glTF/GLB file and populate the engine with its contents.
     *
     * @param engine     Fully initialised engine (after `await engine.init()`).
     * @param url        URL of the .gltf or .glb file.
     * @param rootParent Optional parent node for all top-level glTF scene children.
     *                   When provided, the glTF content is nested under this node,
     *                   allowing the caller to set a root transform (e.g. scale).
     */
    async load(engine: Engine, url: string, rootParent?: NodeHandle): Promise<void> {
        this._log.info(`Loading "${url}" …`);
        this.rootNodeHandle = rootParent;
        const doc = await this._io.read(url);
        const root = doc.getRoot();

        // 1. Textures
        const texMap = await this._loadTextures(engine, root.listTextures());

        // 2. Materials
        const matMap = this._loadMaterials(engine, root.listMaterials(), texMap);

        // 3. Meshes  (one engine MeshHandle per glTF Primitive)
        const primMap = this._loadMeshes(engine, root.listMeshes());

        // 4. Scene nodes
        const nodeMap = new Map<GNode, NodeHandle>();
        const skinList = root.listSkins();
        const defaultScene = root.getDefaultScene() ?? root.listScenes()[0];
        if (defaultScene) {
            for (const child of defaultScene.listChildren()) {
                this._loadNodeTree(engine, child, rootParent, primMap, matMap, nodeMap, skinList);
            }
        }

        // 5. Skins
        this._parseSkins(skinList, nodeMap);

        // 6. Animations
        this._parseAnimations(root.listAnimations(), nodeMap);

        this._log.info(
            `Loaded: ${texMap.size} textures, ${matMap.size} materials, ` +
            `${primMap.size} primitives, ${nodeMap.size} nodes, ` +
            `${this.skins.length} skins, ${this.animations.length} animations`,
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // Textures
    // ──────────────────────────────────────────────────────────────────────

    private async _loadTextures(
        engine: Engine,
        textures: GTexture[],
    ): Promise<Map<GTexture, ResourceHandle>> {
        const map = new Map<GTexture, ResourceHandle>();
        await Promise.all(textures.map(async (tex) => {
            const imageData = tex.getImage();
            if (!imageData) return;

            const mimeType = tex.getMimeType();
            const name     = tex.getName() || tex.getURI() || 'gltf_texture';

            // Color textures (baseColor, emissive) are sRGB; others are linear.
            // We'll upload everything as rgba8unorm and let the material system
            // handle sRGB via the format flag. Normal/MR/AO maps need linear.
            // Since we don't know usage here, we default to rgba8unorm. The
            // material binding picks the correct sampler behavior.
            const handle = await engine.resources.loadBlobToTexture(
                new Blob([imageData], { type: mimeType }),
                { label: name, format: 'rgba8unorm' },
            );
            map.set(tex, handle);
        }));
        return map;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Materials
    // ──────────────────────────────────────────────────────────────────────

    private _loadMaterials(
        engine: Engine,
        materials: GMaterial[],
        texMap: Map<GTexture, ResourceHandle>,
    ): Map<GMaterial, MaterialHandle> {
        const map = new Map<GMaterial, MaterialHandle>();

        for (const mat of materials) {
            const bc = mat.getBaseColorFactor();
            const ef = mat.getEmissiveFactor();

            let alphaMode = AlphaMode.Opaque;
            if (mat.getAlphaMode() === 'MASK')  alphaMode = AlphaMode.Mask;
            if (mat.getAlphaMode() === 'BLEND') alphaMode = AlphaMode.Blend;

            const handle = engine.materials.createMaterial({
                label:       mat.getName() || undefined,
                alphaMode,
                doubleSided: mat.getDoubleSided(),
                pbrParams: {
                    baseColorFactor:   [bc[0], bc[1], bc[2], bc[3]],
                    metallicFactor:    mat.getMetallicFactor(),
                    roughnessFactor:   mat.getRoughnessFactor(),
                    normalScale:       mat.getNormalScale(),
                    occlusionStrength: mat.getOcclusionStrength(),
                    emissiveFactor:    [ef[0], ef[1], ef[2]],
                    alphaCutoff:       mat.getAlphaCutoff(),
                },
                textures: {
                    baseColorMap:         _resolveTex(texMap, mat.getBaseColorTexture()),
                    normalMap:            _resolveTex(texMap, mat.getNormalTexture()),
                    metallicRoughnessMap: _resolveTex(texMap, mat.getMetallicRoughnessTexture()),
                    occlusionMap:         _resolveTex(texMap, mat.getOcclusionTexture()),
                    emissiveMap:          _resolveTex(texMap, mat.getEmissiveTexture()),
                },
            });
            map.set(mat, handle);
        }
        return map;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Meshes
    // ──────────────────────────────────────────────────────────────────────

    /** Each glTF Primitive becomes one engine MeshHandle. */
    private _loadMeshes(
        engine: Engine,
        meshes: GMesh[],
    ): Map<GPrimitive, MeshHandle> {
        const map = new Map<GPrimitive, MeshHandle>();
        for (const mesh of meshes) {
            const prims = mesh.listPrimitives();
            for (let pi = 0; pi < prims.length; pi++) {
                const prim = prims[pi]!;
                const desc = this._buildMeshDescriptor(prim, `${mesh.getName() || 'mesh'}_p${pi}`);
                if (desc) {
                    map.set(prim, engine.meshes.createMesh(desc));
                }
            }
        }
        return map;
    }

    /**
     * Convert a glTF Primitive to the engine's MeshDescriptor.
     * Interleaves position, normal, tangent, uv0 into the 48-byte layout.
     * Computes Mikktspace tangents when the glTF doesn't provide them.
     *
     * When the primitive has JOINTS_0 and WEIGHTS_0 attributes (i.e. it belongs
     * to a skinned mesh), a second vertex buffer is created with the skinning
     * data (stride 24: uint16x4 joints + float32x4 weights).
     */
    private _buildMeshDescriptor(prim: GPrimitive, label: string): MeshDescriptor | null {
        // Only support triangles.
        if (prim.getMode() !== 4 /* TRIANGLES */) {
            this._log.warn(`Skipping non-triangle primitive "${label}" (mode=${prim.getMode()})`);
            return null;
        }

        const posAcc = prim.getAttribute('POSITION');
        if (!posAcc) { this._log.warn(`Primitive "${label}" has no POSITION`); return null; }

        const vertCount = posAcc.getCount();
        const positions = _getFloat32(posAcc);
        const normals   = _getFloat32(prim.getAttribute('NORMAL'));
        const tangents  = _getFloat32(prim.getAttribute('TANGENT'));
        const uvs       = _getFloat32(prim.getAttribute('TEXCOORD_0'));

        // Build index buffer
        let indexData: Uint16Array | Uint32Array | undefined;
        const idxAcc = prim.getIndices();
        if (idxAcc) {
            const raw = idxAcc.getArray();
            if (raw) {
                if (vertCount > 65535) {
                    indexData = Uint32Array.from(raw);
                } else {
                    indexData = new Uint16Array(raw.length);
                    for (let i = 0; i < raw.length; i++) indexData[i] = raw[i]!;
                }
            }
        }

        // Generate flat normals if missing
        const hasNormals = normals !== null && normals.length === vertCount * 3;
        const normArr    = hasNormals ? normals! : _generateFlatNormals(positions!, vertCount, indexData);

        // Generate tangents if missing
        let tangArr: Float32Array;
        if (tangents && tangents.length === vertCount * 4) {
            tangArr = tangents;
        } else {
            const uvArr = uvs && uvs.length === vertCount * 2
                ? uvs
                : new Float32Array(vertCount * 2); // zeros = no UV
            tangArr = _computeTangents(positions!, normArr, uvArr, indexData);
        }

        // UV
        const uvArr = uvs && uvs.length === vertCount * 2 ? uvs : new Float32Array(vertCount * 2);

        // Interleave into 48-byte layout
        const interleaved = new Float32Array(vertCount * FLOATS_PER_VERTEX);
        for (let i = 0; i < vertCount; i++) {
            const o = i * FLOATS_PER_VERTEX;
            // position
            interleaved[o]     = positions![i * 3]!;
            interleaved[o + 1] = positions![i * 3 + 1]!;
            interleaved[o + 2] = positions![i * 3 + 2]!;
            // normal
            interleaved[o + 3] = normArr[i * 3]!;
            interleaved[o + 4] = normArr[i * 3 + 1]!;
            interleaved[o + 5] = normArr[i * 3 + 2]!;
            // tangent (vec4)
            interleaved[o + 6] = tangArr[i * 4]!;
            interleaved[o + 7] = tangArr[i * 4 + 1]!;
            interleaved[o + 8] = tangArr[i * 4 + 2]!;
            interleaved[o + 9] = tangArr[i * 4 + 3]!;
            // uv0
            interleaved[o + 10] = uvArr[i * 2]!;
            interleaved[o + 11] = uvArr[i * 2 + 1]!;
        }

        // Compute AABB
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        for (let i = 0; i < vertCount; i++) {
            const x = positions![i * 3]!, y = positions![i * 3 + 1]!, z = positions![i * 3 + 2]!;
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }
        const aabb: AABB = {
            min: { x: minX, y: minY, z: minZ },
            max: { x: maxX, y: maxY, z: maxZ },
        };

        // Check for skinning attributes (JOINTS_0 + WEIGHTS_0)
        const jointsAcc  = prim.getAttribute('JOINTS_0');
        const weightsAcc = prim.getAttribute('WEIGHTS_0');
        const hasSkinning = jointsAcc !== null && weightsAcc !== null;

        if (hasSkinning) {
            // Build second vertex buffer: uint16x4 joints + float32x4 weights = 24 bytes/vert
            const jointsArr  = jointsAcc!.getArray();
            const weightsArr = _getFloat32(weightsAcc);

            if (jointsArr && weightsArr) {
                // Pack into interleaved skinning buffer
                const skinBuffer = new ArrayBuffer(vertCount * 24);
                const skinView   = new DataView(skinBuffer);

                for (let i = 0; i < vertCount; i++) {
                    const base = i * 24;
                    // JOINTS_0 as uint16x4 (handles both uint8 and uint16 source)
                    skinView.setUint16(base + 0, jointsArr[i * 4]!,     true);
                    skinView.setUint16(base + 2, jointsArr[i * 4 + 1]!, true);
                    skinView.setUint16(base + 4, jointsArr[i * 4 + 2]!, true);
                    skinView.setUint16(base + 6, jointsArr[i * 4 + 3]!, true);
                    // WEIGHTS_0 as float32x4
                    skinView.setFloat32(base + 8,  weightsArr[i * 4]!,     true);
                    skinView.setFloat32(base + 12, weightsArr[i * 4 + 1]!, true);
                    skinView.setFloat32(base + 16, weightsArr[i * 4 + 2]!, true);
                    skinView.setFloat32(base + 20, weightsArr[i * 4 + 3]!, true);
                }

                return {
                    label,
                    vertexLayouts: [INTERLEAVED_LAYOUT, SKINNING_LAYOUT],
                    vertexData:    [interleaved.buffer as ArrayBuffer, skinBuffer],
                    indexData,
                    aabb,
                };
            }
        }

        return {
            label,
            vertexLayouts: [INTERLEAVED_LAYOUT],
            vertexData:    [interleaved.buffer as ArrayBuffer],
            indexData,
            aabb,
        };
    }

    // ──────────────────────────────────────────────────────────────────────
    // Scene nodes
    // ──────────────────────────────────────────────────────────────────────

    private _loadNodeTree(
        engine:   Engine,
        gNode:    GNode,
        parent:   NodeHandle | undefined,
        primMap:  Map<GPrimitive, MeshHandle>,
        matMap:   Map<GMaterial, MaterialHandle>,
        nodeMap:  Map<GNode, NodeHandle>,
        skinList: GSkin[],
    ): void {
        const name   = gNode.getName() || 'node';
        const handle = engine.scene.createNode(name, NodeType.Empty, parent);
        nodeMap.set(gNode, handle);

        // Transform
        const t = gNode.getTranslation();
        const r = gNode.getRotation();
        const s = gNode.getScale();
        engine.scene.setLocalTransform(handle, {
            position: new Float32Array([t[0], t[1], t[2]]),
            rotation: new Float32Array([r[0], r[1], r[2], r[3]]),
            scale:    new Float32Array([s[0], s[1], s[2]]),
        });

        // Mesh
        const gMesh = gNode.getMesh();
        if (gMesh) {
            const prims = gMesh.listPrimitives();
            if (prims.length === 1) {
                // Single primitive — attach directly to this node.
                const mh = primMap.get(prims[0]!);
                if (mh !== undefined) {
                    const gMat = prims[0]!.getMaterial();
                    const matHandle = gMat ? matMap.get(gMat) : undefined;
                    const mats = matHandle !== undefined ? [matHandle] : [];
                    engine.scene.setMeshComponent(handle, mh, mats);
                }
            } else {
                // Multiple primitives — create a child node per primitive.
                for (let pi = 0; pi < prims.length; pi++) {
                    const mh = primMap.get(prims[pi]!);
                    if (mh === undefined) continue;
                    const childHandle = engine.scene.createNode(`${name}_prim${pi}`, NodeType.Mesh, handle);
                    const gMat = prims[pi]!.getMaterial();
                    const matHandle = gMat ? matMap.get(gMat) : undefined;
                    const mats = matHandle !== undefined ? [matHandle] : [];
                    engine.scene.setMeshComponent(childHandle, mh, mats);
                }
            }

            // If this node references a skin, mark it dynamic and record the mapping
            // so the AnimationSystem can link skin → mesh node after loading.
            const gSkin = gNode.getSkin();
            if (gSkin) {
                engine.scene.setNodeStatic(handle, false);
                const skinIdx = skinList.indexOf(gSkin);
                if (skinIdx >= 0) {
                    this.skinNodeMap.set(skinIdx, handle);
                }
            }
        }

        // Recurse children
        for (const child of gNode.listChildren()) {
            this._loadNodeTree(engine, child, handle, primMap, matMap, nodeMap, skinList);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Skins
    // ──────────────────────────────────────────────────────────────────────

    private _parseSkins(
        skins: GSkin[],
        nodeMap: Map<GNode, NodeHandle>,
    ): void {
        for (const skin of skins) {
            const joints = skin.listJoints();
            const jointHandles: NodeHandle[] = [];
            for (const j of joints) {
                const h = nodeMap.get(j);
                if (h !== undefined) jointHandles.push(h);
            }

            let ibm = new Float32Array(joints.length * 16);
            const ibmAcc = skin.getInverseBindMatrices();
            if (ibmAcc) {
                const arr = ibmAcc.getArray();
                if (arr) ibm = Float32Array.from(arr);
            }

            this.skins.push({
                name: skin.getName() || 'skin',
                jointNodeHandles: jointHandles,
                inverseBindMatrices: ibm,
            });
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Animations
    // ──────────────────────────────────────────────────────────────────────

    private _parseAnimations(
        animations: GAnimation[],
        nodeMap: Map<GNode, NodeHandle>,
    ): void {
        for (const anim of animations) {
            const channels: GLTFChannelData[] = [];
            let duration = 0;

            for (const chan of anim.listChannels()) {
                const targetNode = chan.getTargetNode();
                if (!targetNode) continue;
                const nodeHandle = nodeMap.get(targetNode);
                if (nodeHandle === undefined) continue;

                const sampler  = chan.getSampler();
                if (!sampler) continue;
                const inputAcc  = sampler.getInput();
                const outputAcc = sampler.getOutput();
                if (!inputAcc || !outputAcc) continue;

                const inputArr  = inputAcc.getArray();
                const outputArr = outputAcc.getArray();
                if (!inputArr || !outputArr) continue;

                const inputTimes  = Float32Array.from(inputArr);
                const outputValues = Float32Array.from(outputArr);

                const path = chan.getTargetPath() as GLTFChannelData['path'];
                const interpolation = sampler.getInterpolation() as GLTFChannelData['interpolation'];

                if (inputTimes.length > 0) {
                    duration = Math.max(duration, inputTimes[inputTimes.length - 1]!);
                }

                channels.push({
                    targetNodeHandle: nodeHandle,
                    path,
                    interpolation,
                    inputTimes,
                    outputValues,
                });
            }

            this.animations.push({
                name: anim.getName() || 'animation',
                channels,
                duration,
            });
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Module helpers
// ──────────────────────────────────────────────────────────────────────────

function _resolveTex(
    map: Map<GTexture, ResourceHandle>,
    tex: GTexture | null,
): ResourceHandle | undefined {
    if (!tex) return undefined;
    return map.get(tex);
}

/** Get the Float32Array backing an accessor (or null). */
function _getFloat32(acc: GAccessor | null): Float32Array | null {
    if (!acc) return null;
    const arr = acc.getArray();
    if (!arr) return null;
    if (arr instanceof Float32Array) return arr;
    // Convert non-float data (e.g. normalized uint8/uint16) to float.
    const out = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = arr[i]!;
    return out;
}

/**
 * Generate flat (faceted) normals for a triangle mesh that lacks them.
 */
function _generateFlatNormals(
    positions: Float32Array,
    vertCount: number,
    indices?: Uint16Array | Uint32Array,
): Float32Array {
    const normals = new Float32Array(vertCount * 3);
    const triCount = indices ? indices.length / 3 : vertCount / 3;

    for (let t = 0; t < triCount; t++) {
        const i0 = indices ? indices[t * 3]!     : t * 3;
        const i1 = indices ? indices[t * 3 + 1]! : t * 3 + 1;
        const i2 = indices ? indices[t * 3 + 2]! : t * 3 + 2;

        const ax = positions[i1 * 3]!     - positions[i0 * 3]!;
        const ay = positions[i1 * 3 + 1]! - positions[i0 * 3 + 1]!;
        const az = positions[i1 * 3 + 2]! - positions[i0 * 3 + 2]!;
        const bx = positions[i2 * 3]!     - positions[i0 * 3]!;
        const by = positions[i2 * 3 + 1]! - positions[i0 * 3 + 1]!;
        const bz = positions[i2 * 3 + 2]! - positions[i0 * 3 + 2]!;

        let nx = ay * bz - az * by;
        let ny = az * bx - ax * bz;
        let nz = ax * by - ay * bx;
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-10) { nx /= len; ny /= len; nz /= len; }

        for (const idx of [i0, i1, i2]) {
            normals[idx * 3]     += nx;
            normals[idx * 3 + 1] += ny;
            normals[idx * 3 + 2] += nz;
        }
    }

    // Re-normalize accumulated normals
    for (let i = 0; i < vertCount; i++) {
        const x = normals[i * 3]!, y = normals[i * 3 + 1]!, z = normals[i * 3 + 2]!;
        const len = Math.sqrt(x * x + y * y + z * z);
        if (len > 1e-10) {
            normals[i * 3] = x / len; normals[i * 3 + 1] = y / len; normals[i * 3 + 2] = z / len;
        } else {
            normals[i * 3 + 2] = 1; // fallback up
        }
    }
    return normals;
}

/**
 * Mikktspace-compatible tangent generation.
 * Same algorithm as GeometryUtils._computeTangents but works with optional indices.
 */
function _computeTangents(
    positions: Float32Array,
    normals:   Float32Array,
    uvs:       Float32Array,
    indices?:  Uint16Array | Uint32Array,
): Float32Array {
    const vertCount = positions.length / 3;
    const tan1 = new Float32Array(vertCount * 3);
    const tan2 = new Float32Array(vertCount * 3);

    const triCount = indices ? indices.length / 3 : vertCount / 3;
    for (let t = 0; t < triCount; t++) {
        const i0 = indices ? indices[t * 3]!     : t * 3;
        const i1 = indices ? indices[t * 3 + 1]! : t * 3 + 1;
        const i2 = indices ? indices[t * 3 + 2]! : t * 3 + 2;

        const x1 = positions[i1 * 3]!     - positions[i0 * 3]!;
        const y1 = positions[i1 * 3 + 1]! - positions[i0 * 3 + 1]!;
        const z1 = positions[i1 * 3 + 2]! - positions[i0 * 3 + 2]!;
        const x2 = positions[i2 * 3]!     - positions[i0 * 3]!;
        const y2 = positions[i2 * 3 + 1]! - positions[i0 * 3 + 1]!;
        const z2 = positions[i2 * 3 + 2]! - positions[i0 * 3 + 2]!;

        const s1 = uvs[i1 * 2]!     - uvs[i0 * 2]!;
        const t1 = uvs[i1 * 2 + 1]! - uvs[i0 * 2 + 1]!;
        const s2 = uvs[i2 * 2]!     - uvs[i0 * 2]!;
        const t2 = uvs[i2 * 2 + 1]! - uvs[i0 * 2 + 1]!;

        let r = s1 * t2 - s2 * t1;
        if (Math.abs(r) < 1e-10) r = 1.0;
        r = 1.0 / r;

        const sx = (t2 * x1 - t1 * x2) * r;
        const sy = (t2 * y1 - t1 * y2) * r;
        const sz = (t2 * z1 - t1 * z2) * r;
        const tx = (s1 * x2 - s2 * x1) * r;
        const ty = (s1 * y2 - s2 * y1) * r;
        const tz = (s1 * z2 - s2 * z1) * r;

        for (const idx of [i0, i1, i2]) {
            tan1[idx * 3] += sx; tan1[idx * 3 + 1] += sy; tan1[idx * 3 + 2] += sz;
            tan2[idx * 3] += tx; tan2[idx * 3 + 1] += ty; tan2[idx * 3 + 2] += tz;
        }
    }

    const out = new Float32Array(vertCount * 4);
    for (let i = 0; i < vertCount; i++) {
        const nx = normals[i * 3]!, ny = normals[i * 3 + 1]!, nz = normals[i * 3 + 2]!;
        const tx = tan1[i * 3]!,    ty = tan1[i * 3 + 1]!,    tz = tan1[i * 3 + 2]!;
        const bx = tan2[i * 3]!,    by = tan2[i * 3 + 1]!,    bz = tan2[i * 3 + 2]!;

        // Gram-Schmidt orthogonalize
        const dot = nx * tx + ny * ty + nz * tz;
        let ox = tx - nx * dot, oy = ty - ny * dot, oz = tz - nz * dot;
        const len = Math.sqrt(ox * ox + oy * oy + oz * oz);
        if (len > 1e-10) { ox /= len; oy /= len; oz /= len; }

        // Handedness
        const cx = ny * oz - nz * oy;
        const cy = nz * ox - nx * oz;
        const cz = nx * oy - ny * ox;
        const w = (cx * bx + cy * by + cz * bz) < 0.0 ? -1.0 : 1.0;

        out[i * 4] = ox; out[i * 4 + 1] = oy; out[i * 4 + 2] = oz; out[i * 4 + 3] = w;
    }
    return out;
}
