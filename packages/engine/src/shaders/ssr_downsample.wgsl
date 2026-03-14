// /src/shaders/ssr_downsample.wgsl
//
// Gaussian downsample: reads mip N, writes mip N+1 with a weighted 4×4 kernel.
// Used to build the blurred color mip chain for cone-traced glossy SSR.
//
// Bind groups:
//   @group(0) @binding(0) Source mip (texture_2d, read)
//   @group(0) @binding(1) Destination mip (texture_storage_2d, write)

@group(0) @binding(0) var srcMip : texture_2d<f32>;
@group(0) @binding(1) var dstMip : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let dstSize = textureDimensions(dstMip);
    if (gid.x >= dstSize.x || gid.y >= dstSize.y) { return; }

    let srcSize = vec2i(textureDimensions(srcMip));
    // Each destination pixel maps to a 2×2 block in the source.
    // Sample a 4×4 region centered on that block with Gaussian-like weights.
    let base = vec2i(gid.xy) * 2;

    // 4×4 Gaussian weights (approximates σ≈1.0, normalised)
    // Row weights: 1 3 3 1 = 8 per row × 4 rows of same pattern = /64
    var sum = vec4f(0.0);
    let w0 = 1.0 / 36.0;  // corners
    let w1 = 2.0 / 36.0;  // edges
    let w2 = 4.0 / 36.0;  // center 2×2

    for (var dy = -1; dy <= 2; dy++) {
        for (var dx = -1; dx <= 2; dx++) {
            let sc = clamp(base + vec2i(dx, dy), vec2i(0), srcSize - 1);
            let isCenter = (dx >= 0 && dx <= 1 && dy >= 0 && dy <= 1);
            let isEdge   = !isCenter && ((dx >= 0 && dx <= 1) || (dy >= 0 && dy <= 1));
            let w = select(select(w0, w1, isEdge), w2, isCenter);
            sum += textureLoad(srcMip, sc, 0) * w;
        }
    }

    textureStore(dstMip, vec2i(gid.xy), sum);
}
