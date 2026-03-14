// /src/shaders/depth_prepass.wgsl
//
// Depth-only prepass — vertex shader only, no colour output.
//
// Runs before the G-Buffer fill pass to populate the depth buffer with all
// opaque geometry.  The G-Buffer pass then sets depthCompare='equal' and
// disables depth writes, allowing the hardware early-Z unit to discard any
// fragment whose depth differs from the prepass result (i.e. fragments that
// are occluded or were never rasterised here).
//
// Benefits
// ────────
//   • Bandwidth savings — G-Buffer writes are skipped for occluded fragments.
//   • Enables full-resolution depth read-back for SSAO, SSR, Volumetric Fog,
//     Contact Shadows, and per-tile light culling.
//   • Consumed by the DepthDownsample compute pass to build a half-res copy.
//
// Bind groups (layout-compatible subset of gbuffer.wgsl):
//   @group(0) @binding(0)  PerFrameUniforms  — view/proj matrices
//   @group(2) @binding(0)  ModelUniforms     — per-draw model matrix
//
// Group 1 (material) is intentionally absent — depth-only pass needs no
// material parameters.
//
// Vertex layout (stride 48 — same interleaved buffer as the G-Buffer pass):
//   @location(0) position : vec3f  offset  0   ← only attribute consumed here
//   @location(1) normal   : vec3f  offset 12   (present in buffer but ignored)
//   @location(2) tangent  : vec4f  offset 24   (present in buffer but ignored)
//   @location(3) uv0      : vec2f  offset 40   (present in buffer but ignored)

// ── Group 0: Per-frame uniforms ──────────────────────────────────────────────
//
// Mirror of the full PerFrameUniforms struct in gbuffer.wgsl so the
// auto-derived BGL has the same minBindingSize (304 bytes).
// The prepass only reads viewProjectionMatrix; the other fields are
// declared to keep the layout compatible.

struct PerFrameUniforms {
    viewMatrix            : mat4x4f,   //   0 –  63
    projectionMatrix      : mat4x4f,   //  64 – 127
    viewProjectionMatrix  : mat4x4f,   // 128 – 191
    inverseViewProjection : mat4x4f,   // 192 – 255
    cameraPosition        : vec3f,     // 256 – 267
    _pad0                 : f32,       // 268 – 271
    time                  : f32,       // 272 – 275
    deltaTime             : f32,       // 276 – 279
    resolution            : vec2f,     // 280 – 287
    frameIndex            : f32,       // 288 – 291
    exposure              : f32,       // 292 – 295
    jitter                : vec2f,     // 296 – 303
}

@group(0) @binding(0) var<uniform> frame : PerFrameUniforms;

// ── Group 2: Per-draw model matrix (non-instanced) / instance array (instanced) ──
//
// USE_INSTANCING=1  →  storage buffer, indexed by @builtin(instance_index).
// (default)         →  per-draw uniform at a fixed 256-byte ring offset.

#ifdef USE_INSTANCING
@group(2) @binding(0) var<storage, read> instanceMatrices : array<mat4x4f>;
#else
struct ModelUniforms {
    modelMatrix : mat4x4f,
}

@group(2) @binding(0) var<uniform> draw : ModelUniforms;
#endif

#ifdef USE_SKINNING
@group(2) @binding(1) var<storage, read> jointMatrices : array<mat4x4f>;
#endif

// ── Vertex shader ────────────────────────────────────────────────────────────

struct DepthVertexInput {
    @location(0) position : vec3f,
#ifdef USE_SKINNING
    @location(6) joints   : vec4<u32>,
    @location(7) weights  : vec4f,
#endif
}

@vertex
fn vs_main(in: DepthVertexInput, @builtin(instance_index) instanceIdx : u32) -> @builtin(position) vec4f {
#ifdef USE_INSTANCING
    let modelMatrix = instanceMatrices[instanceIdx];
#else
    let modelMatrix = draw.modelMatrix;
#endif

#ifdef USE_SKINNING
    let skinMatrix = in.weights.x * jointMatrices[in.joints.x]
                   + in.weights.y * jointMatrices[in.joints.y]
                   + in.weights.z * jointMatrices[in.joints.z]
                   + in.weights.w * jointMatrices[in.joints.w];
    let worldPos = modelMatrix * (skinMatrix * vec4f(in.position, 1.0));
#else
    let worldPos = modelMatrix * vec4f(in.position, 1.0);
#endif
    return frame.viewProjectionMatrix * worldPos;
}
