// /src/shaders/depth_downsample.wgsl
//
// Compute pass — downsample full-resolution depth32float → half-resolution r32float.
//
// This produces "depth pyramid level 0" (half-res), which is used by:
//   SSR             — screen-space reflection ray marching step size
//   Volumetric Fog  — ray-marched fog depth integration
//   Contact Shadows — short-range shadow ray marching
//   Light Culling   — per-tile depth min/max for clustered/tiled shading
//
// Downsampling strategy: 2×2 gather → store the maximum depth value.
// Maximum depth = farthest from camera in standard [0→1] depth convention,
// which is the most conservative choice for ambient effects (avoids
// over-occlusion by treating partially-covered tiles as fully open).
//
// TODO: extend into a full Hierarchical Z-Buffer (HZB) by chaining additional
//       passes that halve resolution repeatedly down to 1×1.  Each mip level
//       allows ray-march steps proportional to 2^level pixels, dramatically
//       accelerating effects like SSR and volumetric fog.
//
// Bind group 0:
//   binding 0 — srcDepth : texture_depth_2d      (full-res, TEXTURE_BINDING)
//   binding 1 — dstDepth : texture_storage_2d<r32float, write> (half-res, STORAGE_BINDING)

@group(0) @binding(0) var srcDepth : texture_depth_2d;
@group(0) @binding(1) var dstDepth : texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let dstSize = vec2i(textureDimensions(dstDepth));
    let coord   = vec2i(gid.xy);
    if (coord.x >= dstSize.x || coord.y >= dstSize.y) { return; }

    let srcSize = vec2i(textureDimensions(srcDepth, 0));
    let base    = coord * 2;

    // Clamp sample coords to source bounds (handles odd-resolution sources).
    let c00 = clamp(base + vec2i(0, 0), vec2i(0), srcSize - vec2i(1));
    let c10 = clamp(base + vec2i(1, 0), vec2i(0), srcSize - vec2i(1));
    let c01 = clamp(base + vec2i(0, 1), vec2i(0), srcSize - vec2i(1));
    let c11 = clamp(base + vec2i(1, 1), vec2i(0), srcSize - vec2i(1));

    let d00 = textureLoad(srcDepth, c00, 0);
    let d10 = textureLoad(srcDepth, c10, 0);
    let d01 = textureLoad(srcDepth, c01, 0);
    let d11 = textureLoad(srcDepth, c11, 0);

    // Maximum depth = most conservative (farthest from camera).
    let result = max(max(d00, d10), max(d01, d11));
    textureStore(dstDepth, coord, vec4f(result, 0.0, 0.0, 0.0));
}
