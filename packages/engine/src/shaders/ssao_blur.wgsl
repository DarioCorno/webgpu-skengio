// /src/shaders/ssao_blur.wgsl
//
// Depth-aware bilateral blur for half-resolution SSAO.
//
// Single-pass 5x5 Gaussian blur with depth-based edge detection.
// Preserves sharp AO boundaries at depth discontinuities (object edges)
// while smoothing noise from the low sample count.
//
// Bind group 0:
//   binding 0 — aoInput   : texture_2d<f32>  (half-res rgba16float, r=AO, g=linearDepth)
//   binding 1 — params    : SSAOParams (uniform, for blurSharpness)
//   binding 2 — aoOutput  : texture_storage_2d<rgba16float, write> (half-res)

struct SSAOParams {
    radius         : f32,
    bias           : f32,
    intensity      : f32,
    sampleCount    : f32,
    blurSharpness  : f32,
    _pad0          : f32,
    _pad1          : f32,
    _pad2          : f32,
}

@group(0) @binding(0) var aoInput         : texture_2d<f32>;
@group(0) @binding(1) var<uniform> params : SSAOParams;
@group(0) @binding(2) var aoOutput        : texture_storage_2d<rgba16float, write>;

// Precomputed Gaussian weights for 5×5 kernel (sigma ≈ 1.5, exp(-d²/4.5)).
// Row-major indexed by (dy+2)*5 + (dx+2). Avoids exp() per tap.
const KERNEL_RADIUS: i32 = 2;
const GAUSS: array<f32, 25> = array<f32, 25>(
    0.1691, 0.3292, 0.4111, 0.3292, 0.1691,
    0.3292, 0.6412, 0.8007, 0.6412, 0.3292,
    0.4111, 0.8007, 1.0000, 0.8007, 0.4111,
    0.3292, 0.6412, 0.8007, 0.6412, 0.3292,
    0.1691, 0.3292, 0.4111, 0.3292, 0.1691
);

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let size = vec2i(textureDimensions(aoOutput));
    let coord = vec2i(gid.xy);
    if (coord.x >= size.x || coord.y >= size.y) { return; }

    let centerSample = textureLoad(aoInput, coord, 0);
    let centerAO = centerSample.r;
    let centerDepth = centerSample.g; // linear depth stored from compute pass

    if (centerDepth <= 0.0) {
        // Sky pixel — pass through
        textureStore(aoOutput, coord, vec4f(1.0, 0.0, 0.0, 0.0));
        return;
    }

    var totalAO = 0.0;
    var totalWeight = 0.0;

    for (var dy = -KERNEL_RADIUS; dy <= KERNEL_RADIUS; dy++) {
        for (var dx = -KERNEL_RADIUS; dx <= KERNEL_RADIUS; dx++) {
            let sampleCoord = clamp(coord + vec2i(dx, dy), vec2i(0), size - 1);
            let sample = textureLoad(aoInput, sampleCoord, 0);
            let sampleAO = sample.r;
            let sampleDepth = sample.g;

            // Precomputed spatial Gaussian weight (no exp() call)
            let spatialW = GAUSS[(dy + 2) * 5 + (dx + 2)];

            // Depth-aware bilateral weight: fast approx 1/(1+x²) instead of exp(-x²)
            let dd = (centerDepth - sampleDepth) * params.blurSharpness;
            let depthW = 1.0 / (1.0 + dd * dd);

            let w = spatialW * depthW;
            totalAO += sampleAO * w;
            totalWeight += w;
        }
    }

    let blurredAO = totalAO / max(totalWeight, 0.001);
    textureStore(aoOutput, coord, vec4f(blurredAO, 0.0, 0.0, 0.0));
}
