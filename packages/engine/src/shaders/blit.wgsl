// /src/shaders/blit.wgsl
// Fullscreen blit: samples the HDR colour buffer, applies ACES tonemapping
// and approximate sRGB gamma, and writes to the swap-chain.

struct BlitUniforms {
    exposure : f32,
}

@group(0) @binding(0) var colorTex     : texture_2d<f32>;
@group(0) @binding(1) var colorSampler : sampler;
@group(0) @binding(2) var<uniform> blit : BlitUniforms;

// ---- Vertex shader ------------------------------------------------------

struct VertOut {
    @builtin(position) pos : vec4f,
    @location(0)       uv  : vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VertOut {
    // Full-screen triangle — no vertex buffer.
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    // UV: (0,0) = top-left, matching WebGPU texture convention.
    // clip (-1,-1) is bottom-left  → UV (0, 1)
    // clip ( 3,-1)                 → UV (2, 1)   (off-screen, clipped)
    // clip (-1, 3)                 → UV (0, -1)  (off-screen, clipped)
    var uvs = array<vec2f, 3>(
        vec2f(0.0,  1.0),
        vec2f(2.0,  1.0),
        vec2f(0.0, -1.0),
    );
    var out : VertOut;
    out.pos = vec4f(positions[vi], 0.0, 1.0);
    out.uv  = uvs[vi];
    return out;
}

// ---- Tonemapping --------------------------------------------------------

fn acesTonemapping(x: vec3f) -> vec3f {
    // Narkowicz 2015 ACES approximation.
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

// ---- Fragment shader ----------------------------------------------------

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4f {
    let hdr = textureSample(colorTex, colorSampler, in.uv).rgb;

    // Exposure is pre-computed on CPU as a linear multiplier.
    let exposed = hdr * blit.exposure;

    // Tonemap HDR → LDR [0, 1]
    let ldr = acesTonemapping(exposed);

    // Fast sRGB gamma: sqrt ≈ gamma 2.0 (close enough to 2.2)
    let gamma = sqrt(ldr);

    return vec4f(gamma, 1.0);
}
