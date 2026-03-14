// /src/shaders/gbuffer.wgsl
// G-Buffer geometry fill pass — vertex + fragment shader
//
// Bind groups:
//   @group(0) @binding(0)  PerFrameUniforms   (view/proj matrices, camera, time …)
//   @group(1) @binding(0)  MaterialUniforms   (PBR params: baseColor, metallic …)
//   @group(2) @binding(0)  ModelUniforms      (per-draw model matrix)
//
// G-Buffer outputs (three MRTs):
//   RT0  rgba8unorm    albedo RGB  +  occlusion A
//   RT1  rgba16float   world-space normal XYZ (encoded [0,1])  +  roughness A
//   RT2  rgba8unorm    metallic R  +  emissive GBA  (clamped [0,1])
//
// Vertex layout (standard engine interleaved, stride = 48 bytes):
//   @location(0)  position  : vec3f   offset  0
//   @location(1)  normal    : vec3f   offset 12
//   @location(2)  tangent   : vec4f   offset 24  (Mikktspace; w = handedness ±1)
//   @location(3)  uv0       : vec2f   offset 40

// ---- Group 0: Per-frame uniforms -------------------------------------------

struct PerFrameUniforms {
    viewMatrix            : mat4x4f,   //   0 – 63
    projectionMatrix      : mat4x4f,   //  64 – 127
    viewProjectionMatrix  : mat4x4f,   // 128 – 191
    inverseViewProjection : mat4x4f,   // 192 – 255
    cameraPosition        : vec3f,     // 256 – 267
    _pad0                 : f32,       // 268 – 271   (explicit padding)
    time                  : f32,       // 272 – 275
    deltaTime             : f32,       // 276 – 279
    resolution            : vec2f,     // 280 – 287
    frameIndex            : f32,       // 288 – 291
    exposure              : f32,       // 292 – 295
    jitter                : vec2f,     // 296 – 303
}

@group(0) @binding(0) var<uniform> frame : PerFrameUniforms;

// ---- Group 1: Material uniforms --------------------------------------------
//
// Layout matches MaterialSystem.ts MATERIAL_UNIFORM_SIZE = 64 bytes:
//   baseColorFactor   : vec4f   offset  0
//   metallicFactor    : f32     offset 16
//   roughnessFactor   : f32     offset 20
//   normalScale       : f32     offset 24
//   occlusionStrength : f32     offset 28
//   emissiveFactor    : vec3f   offset 32   (vec3f align 16, size 12)
//   alphaCutoff       : f32     offset 44
//   opacity           : f32     offset 48
//   ior               : f32     offset 52
//   _pad              : vec2f   offset 56   (pad to 64)

struct MaterialUniforms {
    baseColorFactor   : vec4f,
    metallicFactor    : f32,
    roughnessFactor   : f32,
    normalScale       : f32,
    occlusionStrength : f32,
    emissiveFactor    : vec3f,
    alphaCutoff       : f32,
    opacity           : f32,
    ior               : f32,
    _pad0             : f32,
    _pad1             : f32,
}

@group(1) @binding(0) var<uniform> material : MaterialUniforms;

// ---- Group 1: Material textures (conditional) ------------------------------
//
// Texture bindings are only declared when the corresponding #define is active.
// A single shared linear sampler at binding 1 covers all texture slots.
//
// Binding layout for @group(1):
//   0  MaterialUniforms     (always)
//   1  materialSampler      (HAS_TEXTURES  — present whenever any map is bound)
//   2  baseColorMap         (HAS_BASE_COLOR_MAP)
//   3  normalMap            (HAS_NORMAL_MAP)
//   4  mrMap                (HAS_MR_MAP  — G=roughness, B=metallic, glTF convention)
//   5  aoMap                (HAS_AO_MAP  — R=occlusion)
//   6  emissiveMap          (HAS_EMISSIVE_MAP)
//
// TODO (Megatexture): When the megatexture system is implemented, bindings 2–6
//   will be replaced by a single large atlas texture bound at a global group
//   (e.g. @group(3) @binding(0)).  Per-material UV offsets and scales will be
//   stored in MaterialUniforms (two vec4f fields: atlasOffsetScale[0..4]).
//   The HAS_*_MAP defines will be superseded by HAS_MEGATEXTURE, and all
//   textureSample() calls below will be replaced by atlas lookup helpers that
//   apply the per-material UV transform before sampling.

#ifdef HAS_TEXTURES
@group(1) @binding(1) var materialSampler : sampler;
#endif

#ifdef HAS_BASE_COLOR_MAP
@group(1) @binding(2) var baseColorMap : texture_2d<f32>;
#endif

#ifdef HAS_NORMAL_MAP
@group(1) @binding(3) var normalMap : texture_2d<f32>;
#endif

#ifdef HAS_MR_MAP
@group(1) @binding(4) var mrMap : texture_2d<f32>;
#endif

#ifdef HAS_AO_MAP
@group(1) @binding(5) var aoMap : texture_2d<f32>;
#endif

#ifdef HAS_EMISSIVE_MAP
@group(1) @binding(6) var emissiveMap : texture_2d<f32>;
#endif

// ---- Group 2: Per-draw model matrix (non-instanced) / instance array (instanced) ---
//
// USE_INSTANCING=1  →  storage buffer indexed by @builtin(instance_index).
//   drawIndexed(indexCount, instanceCount, ..., firstInstance) is issued once
//   per mesh+material batch; the shader reads instanceMatrices[instance_index].
//
// (default)         →  classic per-draw uniform at a fixed offset inside the
//   model-matrix ring buffer; instanceCount is always 1.

#ifdef USE_INSTANCING
@group(2) @binding(0) var<storage, read> instanceMatrices : array<mat4x4f>;
#else
struct ModelUniforms {
    modelMatrix : mat4x4f,
}

@group(2) @binding(0) var<uniform> draw : ModelUniforms;
#endif

// ---- Group 2 binding 1: Joint matrix palette (skeletal animation) ----------
//
// USE_SKINNING=1  →  storage buffer of joint matrices indexed by JOINTS_0.
//   Each vertex blends up to 4 joints using WEIGHTS_0.
//   The joint matrices transform from mesh-local bind pose to mesh-local
//   skinned pose; the model matrix then brings skinned vertices to world space.

#ifdef USE_SKINNING
@group(2) @binding(1) var<storage, read> jointMatrices : array<mat4x4f>;
#endif

// ---- Vertex shader ---------------------------------------------------------

struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) tangent  : vec4f,   // xyz = tangent direction, w = handedness ±1
    @location(3) uv0      : vec2f,
#ifdef USE_SKINNING
    @location(6) joints   : vec4<u32>,   // 4 joint indices (from uint16x4 vertex buffer)
    @location(7) weights  : vec4f,       // 4 blend weights
#endif
}

struct VertexOutput {
    @builtin(position) clipPosition  : vec4f,
    @location(0)       worldPos      : vec3f,
    @location(1)       worldNormal   : vec3f,
    @location(2)       worldTangent  : vec4f,  // xyz = world tangent, w = handedness
    @location(3)       uv0           : vec2f,
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instanceIdx : u32) -> VertexOutput {
#ifdef USE_INSTANCING
    let modelMatrix = instanceMatrices[instanceIdx];
#else
    let modelMatrix = draw.modelMatrix;
#endif

#ifdef USE_SKINNING
    // Compute the skinning matrix by blending up to 4 joint transforms.
    let skinMatrix = in.weights.x * jointMatrices[in.joints.x]
                   + in.weights.y * jointMatrices[in.joints.y]
                   + in.weights.z * jointMatrices[in.joints.z]
                   + in.weights.w * jointMatrices[in.joints.w];

    // Skin in mesh-local space, then transform to world space.
    let skinnedPos     = skinMatrix * vec4f(in.position, 1.0);
    let skinNormalMat  = mat3x3f(skinMatrix[0].xyz, skinMatrix[1].xyz, skinMatrix[2].xyz);
    let skinnedNormal  = skinNormalMat * in.normal;
    let skinnedTangent = skinNormalMat * in.tangent.xyz;

    let worldPos4 = modelMatrix * skinnedPos;
    let m = modelMatrix;
    let normalMat = mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz);
    let worldTangentDir = normalize(normalMat * skinnedTangent);

    var out: VertexOutput;
    out.clipPosition  = frame.viewProjectionMatrix * worldPos4;
    out.worldPos      = worldPos4.xyz;
    out.worldNormal   = normalize(normalMat * skinnedNormal);
    out.worldTangent  = vec4f(worldTangentDir, in.tangent.w);
    out.uv0           = in.uv0;
    return out;
#else
    let worldPos4 = modelMatrix * vec4f(in.position, 1.0);

    // Normal matrix = upper-left 3×3 of the model matrix.
    // Valid for uniform scale and pure rotation; extend to transpose(inverse(M))
    // for non-uniform scale when needed.
    let m = modelMatrix;
    let normalMat = mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz);

    let worldTangentDir = normalize(normalMat * in.tangent.xyz);

    var out: VertexOutput;
    out.clipPosition  = frame.viewProjectionMatrix * worldPos4;
    out.worldPos      = worldPos4.xyz;
    out.worldNormal   = normalize(normalMat * in.normal);
    out.worldTangent  = vec4f(worldTangentDir, in.tangent.w);
    out.uv0           = in.uv0;
    return out;
#endif
}

// ---- Fragment shader -------------------------------------------------------

struct GBufferOutput {
    // RT0 rgba8unorm  — albedo RGB + occlusion A
    @location(0) albedoAO         : vec4f,
    // RT1 rgba16float — world normal XYZ (remapped [0,1]) + roughness W
    @location(1) normalRoughness  : vec4f,
    // RT2 rgba8unorm  — metallic R + emissive GBA (clamped [0,1])
    @location(2) metallicEmissive : vec4f,
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) frontFacing: bool) -> GBufferOutput {

    // ---- Base colour & alpha ------------------------------------------------
#ifdef HAS_BASE_COLOR_MAP
    let baseColorSample = textureSample(baseColorMap, materialSampler, in.uv0);
    let albedo = material.baseColorFactor.rgb * baseColorSample.rgb;
    let alpha  = material.baseColorFactor.a   * baseColorSample.a;
#else
    let albedo = material.baseColorFactor.rgb;
    let alpha  = material.baseColorFactor.a;
#endif

#ifdef ALPHA_MASK
    if (alpha < material.alphaCutoff) { discard; }
#endif

    // ---- World-space normal -------------------------------------------------
    //
    // TBN frame built from the Mikktspace precomputed tangent stored in the
    // vertex buffer (@location(2)).  The bitangent is reconstructed as
    //   B = cross(N, T) * handedness
    // where handedness (±1) is stored in in.worldTangent.w.
    // This avoids the quad-boundary artefacts that the derivative-based approach
    // (dpdx/dpdy) produces at triangle silhouette edges.
#ifdef HAS_NORMAL_MAP
    let Ng   = normalize(in.worldNormal);
    let T    = normalize(in.worldTangent.xyz);
    let B    = cross(Ng, T) * in.worldTangent.w;
    let TBN  = mat3x3f(T, B, Ng);
    let nm   = textureSample(normalMap, materialSampler, in.uv0).rgb * 2.0 - 1.0;
    var N    = normalize(TBN * (nm * vec3f(material.normalScale, material.normalScale, 1.0)));
#else
    var N = normalize(in.worldNormal);
#endif

    // Flip normal for back-facing fragments (double-sided materials only).
#ifdef DOUBLE_SIDED
    if (!frontFacing) { N = -N; }
#endif

    // ---- Metallic / Roughness -----------------------------------------------
    // glTF ORM convention: R=occlusion (ignored here), G=roughness, B=metallic.
#ifdef HAS_MR_MAP
    let mrSample  = textureSample(mrMap, materialSampler, in.uv0);
    let roughness = material.roughnessFactor * mrSample.g;
    let metallic  = material.metallicFactor  * mrSample.b;
#else
    let roughness = material.roughnessFactor;
    let metallic  = material.metallicFactor;
#endif

    // ---- Ambient Occlusion --------------------------------------------------
#ifdef HAS_AO_MAP
    let occlusion = material.occlusionStrength * textureSample(aoMap, materialSampler, in.uv0).r;
#else
    let occlusion = material.occlusionStrength;
#endif

    // ---- Emissive -----------------------------------------------------------
#ifdef HAS_EMISSIVE_MAP
    let emissive = clamp(
        material.emissiveFactor * textureSample(emissiveMap, materialSampler, in.uv0).rgb,
        vec3f(0.0), vec3f(1.0));
#else
    let emissive = clamp(material.emissiveFactor, vec3f(0.0), vec3f(1.0));
#endif

    // ---- Pack G-Buffer outputs ----------------------------------------------
    // Normal remapped [-1,1] → [0,1] for rgba16float storage.
    // Decoded in deferred_lighting.wgsl with: N = normalRough.rgb * 2.0 - 1.0
    let encodedNormal = N * 0.5 + 0.5;

    var out: GBufferOutput;
    out.albedoAO         = vec4f(albedo, occlusion);
    out.normalRoughness  = vec4f(encodedNormal, roughness);
    out.metallicEmissive = vec4f(metallic, emissive);
    return out;
}
