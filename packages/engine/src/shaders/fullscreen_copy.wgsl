// Fullscreen passthrough — samples a texture and writes it unmodified.
// Used to blit one HDR texture into another without tonemapping.
// UVs are interpolated from the vertex shader so the source is always
// sampled across its full [0,1] range regardless of target resolution
// (allows the same pipeline to downsample to a smaller render target).

@group(0) @binding(0) var srcTexture : texture_2d<f32>;
@group(0) @binding(1) var srcSampler : sampler;

struct VsOutput {
    @builtin(position) position : vec4f,
    @location(0)       uv      : vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VsOutput {
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    var out: VsOutput;
    out.position = vec4f(pos[vi], 0.0, 1.0);
    // Map NDC [-1,1] → UV [0,1], flip Y for texture coordinates.
    out.uv = pos[vi] * vec2f(0.5, -0.5) + 0.5;
    return out;
}

@fragment
fn fs_main(in: VsOutput) -> @location(0) vec4f {
    return textureSampleLevel(srcTexture, srcSampler, in.uv, 0.0);
}
