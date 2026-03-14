// /src/shaders/shadow_transparent.wgsl
//
// Shadow map pass for transparent (alpha-blend) materials.
// Includes a fragment shader that discards fragments based on
// opacity * shadowOpacity < 0.5.
//
// Bind groups:
//   @group(0) @binding(0)  lightViewProj : mat4x4f
//   @group(1) @binding(0)  MaterialUniforms (opacity + shadowOpacity)
//   @group(1) @binding(1)  materialSampler  (HAS_TEXTURES)
//   @group(1) @binding(2)  baseColorMap     (HAS_BASE_COLOR_MAP)
//   @group(2) @binding(0)  ModelUniforms (per-draw model matrix)

// ── Group 0: Light view-projection ──────────────────────────────────────

@group(0) @binding(0) var<uniform> lightViewProj : mat4x4f;

// ── Group 1: Material uniforms + optional textures ──────────────────────

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
    shadowOpacity     : f32,
    _pad0             : f32,
}

@group(1) @binding(0) var<uniform> material : MaterialUniforms;

#ifdef HAS_TEXTURES
@group(1) @binding(1) var materialSampler : sampler;
#endif

#ifdef HAS_BASE_COLOR_MAP
@group(1) @binding(2) var baseColorMap : texture_2d<f32>;
#endif

// ── Group 2: Per-draw model matrix ──────────────────────────────────────

struct ModelUniforms {
    modelMatrix : mat4x4f,
}

@group(2) @binding(0) var<uniform> draw : ModelUniforms;

// ── Vertex shader ───────────────────────────────────────────────────────

struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) tangent  : vec4f,
    @location(3) uv0      : vec2f,
}

struct VertexOutput {
    @builtin(position) clipPos : vec4f,
    @location(0)       uv0     : vec2f,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let worldPos = draw.modelMatrix * vec4f(in.position, 1.0);
    var out: VertexOutput;
    out.clipPos = lightViewProj * worldPos;
    out.uv0    = in.uv0;
    return out;
}

// ── Fragment shader ─────────────────────────────────────────────────────

fn hash2d(p: vec2f) -> f32 {
    var p3 = fract(vec3f(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fs_main(in: VertexOutput) -> @builtin(frag_depth) f32 {
    var alpha = material.baseColorFactor.a * material.opacity;

#ifdef HAS_BASE_COLOR_MAP
    alpha *= textureSample(baseColorMap, materialSampler, in.uv0).a;
#endif

    // Discard if surface is fully transparent.
    if (alpha < 0.01) { discard; }

    // Stochastic shadow opacity: randomly discard fragments based on
    // shadowOpacity.  With PCF filtering, this produces a partial shadow
    // whose darkness is proportional to shadowOpacity.
    //   shadowOpacity = 1.0 → no fragments discarded → full shadow
    //   shadowOpacity = 0.3 → ~70% discarded → faint shadow after PCF
    //   shadowOpacity = 0.0 → all discarded → no shadow
    let noise = hash2d(in.clipPos.xy);
    if (noise >= material.shadowOpacity) {
        discard;
    }

    return in.clipPos.z;
}
