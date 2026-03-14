// /src/shaders/shadow.wgsl
//
// Depth-only shadow map pass.
//
// Renders scene geometry into a sub-region of the shadow atlas texture.
// The viewport and scissor rect are set by the CPU to the cascade's atlas region,
// so this shader never needs to know its position within the atlas.
//
// Bind groups:
//   @group(0) @binding(0)  lightViewProj : mat4x4f   — light VP for this cascade/face
//   @group(2) @binding(0)  per-draw ModelUniforms OR instance storage buffer
//
// Group 1 (material) is intentionally absent — depth-only pass needs no material data.
//
// Vertex layout (stride 48 — same interleaved buffer as the G-Buffer pass):
//   @location(0) position : vec3f  offset  0   ← only attribute consumed here
//   (remaining attributes are present in the buffer but ignored)

// ── Group 0: Light view-projection matrix ────────────────────────────────────
//
// Just a single mat4x4f (64 bytes), NOT the full PerFrameUniforms.
// A small dedicated per-cascade uniform buffer is written by the CPU each frame.

@group(0) @binding(0) var<uniform> lightViewProj : mat4x4f;

// ── Group 2: Per-draw model matrix (non-instanced) / instance array (instanced) ──
//
// USE_INSTANCING=1  →  storage buffer indexed by @builtin(instance_index).
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

struct ShadowVertexInput {
    @location(0) position : vec3f,
#ifdef USE_SKINNING
    @location(6) joints   : vec4<u32>,
    @location(7) weights  : vec4f,
#endif
}

@vertex
fn vs_main(
    in: ShadowVertexInput,
    @builtin(instance_index) instanceIdx : u32,
) -> @builtin(position) vec4f {
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
    return lightViewProj * worldPos;
}
