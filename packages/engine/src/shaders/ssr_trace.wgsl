// /src/shaders/ssr_trace.wgsl
//
// Screen-space reflections — DDA ray march (compute shader).
//
// Uses perspective-correct screen-space stepping based on the DDA line
// algorithm (Morgan McGuire / Mike Mara).  For each pixel that needs
// reflections, the shader projects a reflection ray to screen space and
// marches along the major axis, testing against the depth buffer with a
// linearised thickness comparison.
//
// Output: rgba16float texture
//   rg = hit UV (normalised [0,1])
//   b  = confidence (0 = miss, 1 = solid hit)
//   a  = hit distance in view-space (for cone tracing mip selection)
//
// Bind groups:
//   @group(0) @binding(0) PerFrameUniforms
//   @group(0) @binding(1) G-Buffer normal+roughness (rgba16float)
//   @group(0) @binding(2) Full-res depth (texture_depth_2d)
//   @group(0) @binding(3) SSR params uniform
//   @group(0) @binding(4) Output SSR texture (rgba16float, write)

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

struct SSRParams {
    maxRaySteps     : f32,
    thickness       : f32,
    stride          : f32,
    fadeEnd         : f32,
    roughnessCutoff : f32,
    jitterScale     : f32,
    maxDistance      : f32,
    strideZCutoff   : f32,
    _pad0           : f32,
    _pad1           : f32,
    _pad2           : f32,
    _pad3           : f32,
}

@group(0) @binding(0) var<uniform> frame     : PerFrameUniforms;
@group(0) @binding(1) var gbNormalRoughness   : texture_2d<f32>;
@group(0) @binding(2) var gbDepth             : texture_depth_2d;
@group(0) @binding(3) var<uniform> params     : SSRParams;
@group(0) @binding(4) var ssrOutput           : texture_storage_2d<rgba16float, write>;

// ── Helpers ─────────────────────────────────────────────────────────────

fn reconstructWorldPos(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec4f(uv * 2.0 - 1.0, depth, 1.0);
    let ndcFlipped = vec4f(ndc.x, -ndc.y, ndc.z, 1.0);
    let world = frame.inverseViewProjection * ndcFlipped;
    return world.xyz / world.w;
}

fn linearizeDepth(ndcZ: f32) -> f32 {
    return frame.projectionMatrix[3][2] / (ndcZ + frame.projectionMatrix[2][2]);
}

fn hash(p: vec2f) -> f32 {
    var p3 = fract(vec3f(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn projectToScreen(worldPos: vec3f) -> vec4f {
    let clip = frame.viewProjectionMatrix * vec4f(worldPos, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    return vec4f(uv, ndc.z, 1.0 / clip.w);
}

fn intersectsDepthBuffer(sceneZ: f32, rayZMin: f32, rayZMax: f32, thickness: f32) -> bool {
    return (rayZMax >= sceneZ) && (rayZMin - thickness <= sceneZ);
}

// ── Main ────────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let outputSize = textureDimensions(ssrOutput);
    if (gid.x >= outputSize.x || gid.y >= outputSize.y) { return; }

    let coord   = vec2i(gid.xy);
    let texSize = vec2f(outputSize);
    let uv      = (vec2f(gid.xy) + 0.5) / texSize;

    // Sample depth — skip sky pixels
    let depth = textureLoad(gbDepth, coord, 0);
    if (depth >= 1.0) {
        textureStore(ssrOutput, coord, vec4f(0.0));
        return;
    }

    // Decode G-Buffer
    let normalRough = textureLoad(gbNormalRoughness, coord, 0);
    let rawNormal   = normalRough.rgb * 2.0 - 1.0;
    let N           = normalize(rawNormal);
    let roughness   = normalRough.a;

    if (roughness > params.roughnessCutoff) {
        textureStore(ssrOutput, coord, vec4f(0.0));
        return;
    }

    // Reconstruct world position and compute reflection
    let worldPos = reconstructWorldPos(uv, depth);
    let V        = normalize(frame.cameraPosition - worldPos);
    let R        = reflect(-V, N);

    // Offset origin along normal
    let csOrig = worldPos + N * 0.01;

    // ── Project ray to screen space ─────────────────────────────────────
    let rayEnd = csOrig + R * params.maxDistance;

    let p0 = projectToScreen(csOrig);   // xy=uv, z=ndcZ, w=1/clipW
    let p1 = projectToScreen(rayEnd);

    // Pixel coordinates
    let pp0 = p0.xy * texSize;
    let pp1 = p1.xy * texSize;

    // DDA setup: step along the major axis (whichever spans more pixels)
    var dP = pp1 - pp0;
    let absDp = abs(dP);
    let stepDir = select(-1.0, 1.0, dP.x > 0.0 || (dP.x == 0.0 && dP.y > 0.0));

    // Swap so we always step along the major axis
    let isXMajor = absDp.x >= absDp.y;
    let majorLen = select(absDp.y, absDp.x, isXMajor);

    if (majorLen < 0.001) {
        textureStore(ssrOutput, coord, vec4f(0.0));
        return;
    }

    // Perspective-correct interpolation: interpolate Q=z/w and K=1/w linearly
    let Q0 = p0.z * p0.w;  // ndcZ / clipW  = z/w but p0.w is already 1/clipW
    let Q1 = p1.z * p1.w;  // so Q = ndcZ * (1/clipW) -- but ndcZ = clipZ/clipW, so Q = clipZ/clipW²
    // Actually for perspective correct: we interpolate (z/w, 1/w) then recover z = (z/w)/(1/w)
    // p0.z = ndcZ0 = clipZ0/clipW0, p0.w = 1/clipW0
    // So z/w = p0.z * p0.w = clipZ0/clipW0², 1/w = p0.w = 1/clipW0
    // Recovered ndcZ = (z/w) / (1/w) = clipZ0/clipW0 = p0.z ✓

    // Normalise step to 1 pixel on the major axis
    let invMajor = 1.0 / majorLen;
    let dPstep = dP * invMajor;
    let dQ = (Q1 - Q0) * invMajor;
    let dK = (p1.w - p0.w) * invMajor;

    // Apply stride: adaptive based on view-space depth
    let viewDepth = linearizeDepth(depth);
    let strideScale = 1.0 - min(1.0, viewDepth * params.strideZCutoff);
    let stride = max(1.0, params.stride * (1.0 + strideScale));

    let dPs = dPstep * stride;
    let dQs = dQ * stride;
    let dKs = dK * stride;

    // Jitter start position for temporal stability
    let jitter = hash(vec2f(gid.xy) + frame.frameIndex * 1.618034) * params.jitterScale;

    var PQK = vec3f(Q0, p0.w, 0.0);  // Q, K, unused
    var P   = pp0;
    // Advance by jitter
    P   += dPs * jitter;
    PQK += vec3f(dQs, dKs, 0.0) * jitter;

    // ── DDA ray march ───────────────────────────────────────────────────
    let maxSteps = i32(params.maxRaySteps);
    var hit      = false;
    var hitUV    = vec2f(0.0);
    var hitDist  = 0.0;

    var prevZMaxEstimate = linearizeDepth(PQK.x / PQK.y);
    var rayZMin = prevZMaxEstimate;
    var rayZMax = prevZMaxEstimate;

    for (var i = 0; i < maxSteps; i++) {
        P   += dPs;
        PQK += vec3f(dQs, dKs, 0.0);

        // Recover NDC Z from perspective-correct interpolation
        let currentNdcZ = PQK.x / PQK.y;

        rayZMin = prevZMaxEstimate;
        rayZMax = linearizeDepth(currentNdcZ);

        // Ensure min <= max
        if (rayZMin > rayZMax) {
            let tmp = rayZMin;
            rayZMin = rayZMax;
            rayZMax = tmp;
        }

        prevZMaxEstimate = rayZMax;

        // Convert pixel position back to UV
        let sampleUV = P / texSize;

        // Out of screen bounds
        if (sampleUV.x <= 0.0 || sampleUV.x >= 1.0 || sampleUV.y <= 0.0 || sampleUV.y >= 1.0) {
            break;
        }

        let sampleCoord  = vec2i(clamp(sampleUV, vec2f(0.0), vec2f(0.99999)) * texSize);
        let sampledDepth = textureLoad(gbDepth, sampleCoord, 0);
        let sceneZ       = linearizeDepth(sampledDepth);

        if (sceneZ < 0.001) { continue; } // skip empty depth

        if (intersectsDepthBuffer(sceneZ, rayZMin, rayZMax, params.thickness)) {
            hit   = true;
            hitUV = sampleUV;
            break;
        }
    }

    // ── Backface rejection ─────────────────────────────────────────────
    if (hit) {
        let hitCoord   = vec2i(clamp(hitUV, vec2f(0.0), vec2f(0.99999)) * texSize);
        let hitNormRaw = textureLoad(gbNormalRoughness, hitCoord, 0).rgb * 2.0 - 1.0;
        let hitN       = normalize(hitNormRaw);
        if (dot(R, hitN) > 0.0) {
            hit = false;
        }
    }

    // ── Binary refinement ──────────────────────────────────────────────
    if (hit) {
        // Step back half and refine
        var rP   = P - dPs * 0.5;
        var rPQK = PQK - vec3f(dQs, dKs, 0.0) * 0.5;
        var sP   = dPs * 0.25;
        var sPQK = vec3f(dQs, dKs, 0.0) * 0.25;

        for (var r = 0; r < 5; r++) {
            let rUV = rP / texSize;
            let rCoord = vec2i(clamp(rUV, vec2f(0.0), vec2f(0.99999)) * texSize);
            let sd = textureLoad(gbDepth, rCoord, 0);
            let rNdcZ = rPQK.x / rPQK.y;
            let diff = linearizeDepth(rNdcZ) - linearizeDepth(sd);
            if (diff > 0.0) {
                rP   -= sP;
                rPQK -= sPQK;
            } else {
                rP   += sP;
                rPQK += sPQK;
            }
            sP   *= 0.5;
            sPQK *= 0.5;
        }
        hitUV = rP / texSize;
    }

    if (!hit) {
        textureStore(ssrOutput, coord, vec4f(0.0));
        return;
    }

    // ── Confidence fades ───────────────────────────────────────────────
    // Screen edge fade
    let boundary = abs(hitUV - 0.5) * 2.0;
    let edgeFade = 1.0 - smoothstep(params.fadeEnd, 1.0, max(boundary.x, boundary.y));

    // Roughness fade
    let roughFade = 1.0 - smoothstep(params.roughnessCutoff * 0.8, params.roughnessCutoff, roughness);

    // Distance fade
    let hitWorldPos = reconstructWorldPos(hitUV, textureLoad(gbDepth, vec2i(clamp(hitUV, vec2f(0.0), vec2f(0.99999)) * texSize), 0));
    hitDist = length(hitWorldPos - worldPos);
    let distFade = 1.0 - smoothstep(params.maxDistance * 0.7, params.maxDistance, hitDist);

    let confidence = edgeFade * roughFade * distFade;

    textureStore(ssrOutput, coord, vec4f(hitUV, confidence, hitDist));
}
