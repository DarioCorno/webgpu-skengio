// /src/shaders/ssao_composite.wgsl
//
// Fullscreen composite pass — multiplies the blurred half-res AO
// into the full-res HDR scene colour.
//
// Bind group 0:
//   binding 0 — hdrColor   : texture_2d<f32>  (full-res scene colour)
//   binding 1 — aoBlurred  : texture_2d<f32>  (half-res blurred AO, r channel)
//   binding 2 — linearSamp : sampler          (linear filtering for bilinear upscale)

@group(0) @binding(0) var hdrColor   : texture_2d<f32>;
@group(0) @binding(1) var aoBlurred  : texture_2d<f32>;
@group(0) @binding(2) var linearSamp : sampler;

struct VSOut {
    @builtin(position) position : vec4f,
    @location(0)       uv       : vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    // Fullscreen triangle (3 vertices, no vertex buffer)
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VSOut;
    out.position = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    let fullSize = vec2i(textureDimensions(hdrColor, 0));
    let coord = vec2i(in.position.xy);
    let color = textureLoad(hdrColor, coord, 0);

    // Bilinear sample the half-res AO using the linear sampler
    let ao = textureSample(aoBlurred, linearSamp, in.uv).r;

    return vec4f(color.rgb * ao, color.a);
}
