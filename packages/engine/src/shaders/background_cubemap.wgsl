// /src/shaders/background_cubemap.wgsl
//
// Fullscreen cubemap background that rotates with the camera.
// Rendered with depthCompare='equal' at depth=1.0 so it only fills sky pixels.

struct BgUniforms {
    bgType     : u32,       // (unused, always 3 for cubemap)
    _pad0      : u32,
    _pad1      : u32,
    _pad2      : u32,
    color      : vec4f,     // (unused)
    topColor   : vec4f,     // (unused)
    invViewProj: mat4x4f,   // inverse view-projection for ray direction
}

@group(0) @binding(0) var<uniform> bg      : BgUniforms;
@group(0) @binding(1) var          cubeTex : texture_cube<f32>;
@group(0) @binding(2) var          cubeSamp: sampler;

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

// ---- Fragment shader (cubemap lookup) -----------------------------------

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4f {
    // Reconstruct world-space ray direction from clip coords via inverse VP.
    // UV (0,0)=top-left, (1,1)=bottom-right. Map to NDC: x∈[-1,1], y∈[-1,1].
    let ndcXY = in.uv * 2.0 - 1.0;
    let ndc = vec4f(ndcXY.x, -ndcXY.y, 1.0, 1.0);
    let worldPos = bg.invViewProj * ndc;
    let dir = normalize(worldPos.xyz / worldPos.w);

    let col = textureSample(cubeTex, cubeSamp, dir);
    return vec4f(col.rgb, 1.0);
}
