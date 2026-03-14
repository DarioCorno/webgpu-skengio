// /src/shaders/ssao_compute.wgsl
//
// Half-resolution SSAO compute shader.
//
// Reads the half-res depth (r32float) and full-res G-Buffer normals,
// computes ambient occlusion using hemisphere sampling with a golden-angle
// spiral pattern, outputs AO to a half-res rgba16float texture.
//
// Bind group 0:
//   binding 0 — frame        : PerFrameUniforms (uniform)
//   binding 1 — halfDepth    : texture_2d<f32>  (half-res r32float)
//   binding 2 — gbNormals    : texture_2d<f32>  (full-res rgba16float, normal+roughness)
//   binding 3 — params       : SSAOParams (uniform)
//   binding 4 — aoOutput     : texture_storage_2d<rgba16float, write> (half-res)
//   binding 5 — sampleKernel : array<vec4f> (precomputed hemisphere samples, storage)

struct PerFrameUniforms {
    viewMatrix            : mat4x4f,
    projectionMatrix      : mat4x4f,
    viewProjectionMatrix  : mat4x4f,
    inverseViewProjection : mat4x4f,
    cameraPosition        : vec3f,
    _pad0                 : f32,
    time                  : f32,
    deltaTime             : f32,
    resolution            : vec2f,
    frameIndex            : f32,
    exposure              : f32,
    jitter                : vec2f,
}

struct SSAOParams {
    radius         : f32,   // world-space hemisphere radius
    bias           : f32,   // depth bias to avoid self-occlusion
    intensity      : f32,   // AO strength multiplier
    sampleCount    : f32,   // number of samples (cast to u32)
    blurSharpness  : f32,   // depth-edge sensitivity for blur pass
    _pad0          : f32,
    _pad1          : f32,
    _pad2          : f32,
}

@group(0) @binding(0) var<uniform> frame  : PerFrameUniforms;
@group(0) @binding(1) var halfDepth       : texture_2d<f32>;
@group(0) @binding(2) var gbNormals       : texture_2d<f32>;
@group(0) @binding(3) var<uniform> params : SSAOParams;
@group(0) @binding(4) var aoOutput        : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<storage, read> sampleKernel : array<vec4f>;

// Simple hash for per-pixel rotation
fn hash(p: vec2u) -> f32 {
    var n = p.x * 1597u + p.y * 2753u + 1013u;
    n = (n << 13u) ^ n;
    n = n * (n * n * 15731u + 789221u) + 1376312589u;
    return f32(n & 0x7fffffffu) / f32(0x7fffffff);
}

// Build a jitter-free projection matrix for stable SSAO.
// TAA jitter is added to projectionMatrix[2][0] and [2][1] (column 2, rows 0/1).
// We subtract it so reconstructed positions don't shift each frame.
fn getStableProj() -> mat4x4f {
    var proj = frame.projectionMatrix;
    proj[2][0] -= frame.jitter.x;
    proj[2][1] -= frame.jitter.y;
    return proj;
}

fn linearizeDepth(d: f32, proj: mat4x4f) -> f32 {
    let projA = proj[2][2];
    let projB = proj[3][2];
    return projB / (d + projA);
}

fn reconstructViewPos(uv: vec2f, linearZ: f32, proj: mat4x4f) -> vec3f {
    let ndcX = uv.x * 2.0 - 1.0;
    let ndcY = 1.0 - uv.y * 2.0;
    let viewX = ndcX * linearZ / proj[0][0];
    let viewY = ndcY * linearZ / proj[1][1];
    return vec3f(viewX, viewY, -linearZ);
}

// Build TBN from normal and orient a hemisphere sample
fn orientSample(s: vec3f, n: vec3f) -> vec3f {
    let up = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(n.z) < 0.99);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return t * s.x + b * s.y + n * s.z;
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let halfSize = vec2i(textureDimensions(aoOutput));
    let coord = vec2i(gid.xy);
    if (coord.x >= halfSize.x || coord.y >= halfSize.y) { return; }

    // Read half-res depth
    let depth = textureLoad(halfDepth, coord, 0).r;
    if (depth >= 1.0) {
        // Sky — no occlusion
        textureStore(aoOutput, coord, vec4f(1.0, 0.0, 0.0, 0.0));
        return;
    }

    let proj = getStableProj();
    let linearZ = linearizeDepth(depth, proj);
    let uv = (vec2f(coord) + 0.5) / vec2f(halfSize);
    let viewPos = reconstructViewPos(uv, linearZ, proj);

    // Read world-space normal from full-res G-Buffer, convert to view space
    let fullCoord = clamp(coord * 2, vec2i(0), vec2i(textureDimensions(gbNormals, 0)) - 1);
    let rawNormal = textureLoad(gbNormals, fullCoord, 0).rgb * 2.0 - 1.0;
    let viewN = normalize((frame.viewMatrix * vec4f(rawNormal, 0.0)).xyz);

    // Per-pixel rotation via 2D rotation of sample XY
    let rotAngle = hash(vec2u(gid.xy)) * 6.283185;
    let cosRot = cos(rotAngle);
    let sinRot = sin(rotAngle);

    let numSamples = u32(params.sampleCount);
    var ao = 0.0;

    for (var i = 0u; i < numSamples; i++) {
        // Read precomputed hemisphere sample (xyz = normalized direction, w = radius weight)
        let kern = sampleKernel[i];
        // Rotate sample XY by per-pixel angle for spatial noise
        let rx = kern.x * cosRot - kern.y * sinRot;
        let ry = kern.x * sinRot + kern.y * cosRot;
        let sampleDir = vec3f(rx, ry, kern.z);
        let r = kern.w; // radius weighting factor

        // Orient to surface normal and scale by radius
        let offset = orientSample(sampleDir, viewN) * params.radius * r;
        let samplePos = viewPos + offset;

        // Project sample position to screen UV (jitter-free)
        let clipPos = proj * vec4f(samplePos, 1.0);
        let ndcXY = clipPos.xy / clipPos.w;
        let sampleUV = vec2f(ndcXY.x * 0.5 + 0.5, 0.5 - ndcXY.y * 0.5);

        // Read depth at sample UV (half-res coordinates)
        let sampleCoord = vec2i(sampleUV * vec2f(halfSize));
        let clampedCoord = clamp(sampleCoord, vec2i(0), halfSize - 1);

        // Check bounds — skip samples that project outside the screen
        if (sampleCoord.x < 0 || sampleCoord.x >= halfSize.x ||
            sampleCoord.y < 0 || sampleCoord.y >= halfSize.y) {
            continue;
        }

        let sampleDepth = textureLoad(halfDepth, clampedCoord, 0).r;
        let sampleLinearZ = linearizeDepth(sampleDepth, proj);

        // Is the geometry at this sample closer than our sample point?
        // samplePos.z is negative (view space), so -samplePos.z = distance from camera
        let sampleViewZ = -samplePos.z;
        let depthDiff = sampleLinearZ - sampleViewZ;

        // Soft occlusion (smoothstep instead of hard step to avoid flickering)
        // Occluded when depthDiff < -bias (geometry closer than sample point)
        let occlusion = 1.0 - smoothstep(-params.bias * 2.0, -params.bias * 0.5, depthDiff);

        // Range check: fade out contributions from geometry too far away
        let rangeCheck = smoothstep(params.radius, params.radius * 0.5, abs(depthDiff));
        ao += occlusion * rangeCheck;
    }

    ao = 1.0 - (ao / f32(numSamples)) * params.intensity;
    ao = clamp(ao, 0.0, 1.0);

    textureStore(aoOutput, coord, vec4f(ao, linearZ, 0.0, 0.0));
}
