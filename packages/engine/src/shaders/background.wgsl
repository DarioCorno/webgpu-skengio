// /src/shaders/background.wgsl
//
// Fullscreen background: solid colour, vertical gradient, or 2D texture.
// Rendered with depthCompare='equal' at depth=1.0 so it only fills sky pixels.

struct BgUniforms {
    bgType     : u32,       // 0=color, 1=gradient, 2=texture
    _pad0      : u32,
    _pad1      : u32,
    _pad2      : u32,
    color      : vec4f,     // solid colour  OR  gradient bottom colour
    topColor   : vec4f,     // gradient top colour
    invViewProj: mat4x4f,   // (unused here, kept for layout compat)
}

@group(0) @binding(0) var<uniform> bg      : BgUniforms;
@group(0) @binding(1) var          bgTex   : texture_2d<f32>;
@group(0) @binding(2) var          bgSamp  : sampler;

// ---- Vertex shader (fullscreen triangle) --------------------------------

struct VertOut {
    @builtin(position) pos : vec4f,
    @location(0)       uv  : vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VertOut {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    var uvs = array<vec2f, 3>(
        vec2f(0.0,  1.0),
        vec2f(2.0,  1.0),
        vec2f(0.0, -1.0),
    );
    var out : VertOut;
    out.pos = vec4f(positions[vi], 1.0, 1.0);  // z=1.0 → far plane
    out.uv  = uvs[vi];
    return out;
}

// ---- Fragment shader (color / gradient / texture) -----------------------

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4f {
    let t = bg.bgType;

    if (t == 0u) {
        // Solid colour
        return vec4f(bg.color.rgb, 1.0);
    }
    if (t == 1u) {
        // Vertical gradient: bottom colour at UV.y=1, top colour at UV.y=0
        let blend = in.uv.y;   // 0 at top, 1 at bottom
        let col = mix(bg.topColor.rgb, bg.color.rgb, blend);
        return vec4f(col, 1.0);
    }
    // t == 2: 2D texture
    let texCol = textureSample(bgTex, bgSamp, in.uv);
    return vec4f(texCol.rgb, 1.0);
}
