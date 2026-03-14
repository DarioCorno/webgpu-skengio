// /src/shaders/ssr_composite.wgsl
//
// Fullscreen pass: cone-traced glossy SSR composite with cubemap fallback.
//
// Reads the SSR trace texture (rg=hitUV, b=confidence, a=hitDistance)
// and samples a pre-blurred colour mip chain at the appropriate mip level
// based on roughness and hit distance (cone tracing).
//
// For misses or partial hits, falls back to the environment cubemap
// sampled along the reflection direction.
//
// Bind groups:
//   @group(0) @binding(0) HDR colour input (texture_2d<f32>)
//   @group(0) @binding(1) SSR trace result (texture_2d<f32>)
//   @group(0) @binding(2) Blurred colour mip chain (texture_2d<f32>, N mips)
//   @group(0) @binding(3) G-Buffer normal+roughness (texture_2d<f32>)
//   @group(0) @binding(4) G-Buffer metallic+emissive (texture_2d<f32>)
//   @group(0) @binding(5) Depth (texture_depth_2d)
//   @group(0) @binding(6) Environment cubemap (texture_cube<f32>)
//   @group(0) @binding(7) Linear sampler (with mip filtering)
//   @group(0) @binding(8) PerFrameUniforms
//   @group(0) @binding(9) SSR composite params

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

struct SSRCompositeParams {
    maxMipLevel    : f32,
    envFallbackStr : f32,
    envCubemapEnabled : f32,
    _pad           : f32,
}

@group(0) @binding(0) var hdrColor          : texture_2d<f32>;
@group(0) @binding(1) var ssrTrace          : texture_2d<f32>;
@group(0) @binding(2) var colorMipChain     : texture_2d<f32>;
@group(0) @binding(3) var gbNormalRoughness : texture_2d<f32>;
@group(0) @binding(4) var gbMetallicEmissive: texture_2d<f32>;
@group(0) @binding(5) var gbDepth           : texture_depth_2d;
@group(0) @binding(6) var envCubemap        : texture_cube<f32>;
@group(0) @binding(7) var linearSamp        : sampler;
@group(0) @binding(8) var<uniform> frame    : PerFrameUniforms;
@group(0) @binding(9) var<uniform> compParams : SSRCompositeParams;

// ── Helpers ─────────────────────────────────────────────────────────────

fn reconstructWorldPos(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec4f(uv * 2.0 - 1.0, depth, 1.0);
    let ndcFlipped = vec4f(ndc.x, -ndc.y, ndc.z, 1.0);
    let world = frame.inverseViewProjection * ndcFlipped;
    return world.xyz / world.w;
}

// Isosceles triangle helpers for cone tracing (from willpgfx article)
fn isoscelesTriangleOpposite(adjacentLen: f32, coneHalfAngle: f32) -> f32 {
    return 2.0 * tan(coneHalfAngle) * adjacentLen;
}

fn isoscelesTriangleInRadius(a: f32, h: f32) -> f32 {
    // Inscribed circle radius of isosceles triangle
    // a = opposite (base) length, h = adjacent (height) length
    let a2 = a * a;
    let fh2 = 4.0 * h * h;
    return (a * (sqrt(a2 + fh2) - a)) / (4.0 * h);
}

fn isoscelesTriangleNextAdjacent(adjacentLen: f32, inRadius: f32) -> f32 {
    return adjacentLen - inRadius * 2.0;
}

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
    out.pos = vec4f(positions[vi], 0.0, 1.0);
    out.uv  = uvs[vi];
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4f {
    let sceneColor = textureSample(hdrColor, linearSamp, in.uv);

    // Read G-Buffer (all texture reads before any non-uniform branching)
    let normalRough   = textureSample(gbNormalRoughness, linearSamp, in.uv);
    let rawNormal     = normalRough.rgb * 2.0 - 1.0;
    let N             = normalize(rawNormal);
    let roughness     = normalRough.a;

    let metalEmissive = textureSample(gbMetallicEmissive, linearSamp, in.uv);
    let metallic      = metalEmissive.r;

    // All textureSample calls must happen before any non-uniform branching
    let ssrData    = textureSample(ssrTrace, linearSamp, in.uv);

    // Depth is texture_depth_2d — must use textureLoad (not compatible with filtering sampler)
    let pixCoord   = vec2i(floor(in.uv * frame.resolution));
    let depth      = textureLoad(gbDepth, pixCoord, 0);

    // Reconstruct world position for reflection direction and Fresnel
    let worldPos = reconstructWorldPos(in.uv, depth);
    let V        = normalize(frame.cameraPosition - worldPos);
    let R        = reflect(-V, N);

    // Fresnel (Schlick)
    let F0     = mix(vec3f(0.04), vec3f(1.0), metallic);
    let NdotV  = saturate(dot(N, V));
    let ft = 1.0 - NdotV; let ft2 = ft * ft;
    let fresnel = F0 + (vec3f(1.0) - F0) * (ft2 * ft2 * ft);

    // Unpack SSR trace result
    let hitUV      = clamp(ssrData.rg, vec2f(0.0), vec2f(1.0));
    let confidence = saturate(ssrData.b);
    let hitDist    = ssrData.a;

    // ── Cone-traced glossy sampling ─────────────────────────────────────
    // Convert roughness to a cone half-angle (GGX lobe approximation)
    let coneHalfAngle = roughness * roughness * 0.7854; // PI/4 ≈ 0.7854

    var ssrColor = vec3f(0.0);

    if (confidence > 0.001) {
        // Screen-space distance from origin to hit
        let texSize       = vec2f(textureDimensions(hdrColor));
        let originPx      = in.uv * texSize;
        let hitPx         = hitUV * texSize;
        let adjacentLen   = length(hitPx - originPx);
        let adjacentUnit  = select(vec2f(1.0, 0.0), normalize(hitPx - originPx), adjacentLen > 0.001);

        if (coneHalfAngle < 0.001) {
            // Mirror reflection: sample from the sharp HDR color directly
            ssrColor = textureSampleLevel(hdrColor, linearSamp, hitUV, 0.0).rgb;
        } else {
            // Cone trace: walk inscribed circles along the triangle
            var totalColor = vec4f(0.0);  // rgb + accumulated weight
            var adjLen     = adjacentLen;
            var glossMult  = 1.0 - roughness;  // compound gloss weight

            let maxScreenDim = max(texSize.x, texSize.y);

            for (var i = 0; i < 7; i++) {
                if (adjLen <= 0.0 || totalColor.a >= 1.0) { break; }

                let oppositeLen = isoscelesTriangleOpposite(adjLen, coneHalfAngle);
                let incircleR   = isoscelesTriangleInRadius(oppositeLen, adjLen);

                // Sample position: along the adjacent axis, offset by incircle radius from tip
                let samplePx = originPx + adjacentUnit * max(adjLen - incircleR, 0.0);
                let sampleUV = clamp(samplePx / texSize, vec2f(0.0), vec2f(1.0));

                // Mip level from incircle diameter in pixels.
                // colorMipChain mip 0 = half-res blur, so rawMip maps directly.
                let mipLevel = clamp(log2(max(incircleR * 2.0, 1.0)), 0.0, compParams.maxMipLevel);

                let sampleColor = textureSampleLevel(colorMipChain, linearSamp, sampleUV, mipLevel).rgb;

                let weight = glossMult;
                totalColor += vec4f(sampleColor * weight, weight);

                // Advance to next inscribed circle
                adjLen = isoscelesTriangleNextAdjacent(adjLen, incircleR);
                glossMult *= (1.0 - roughness);
            }

            if (totalColor.a > 0.001) {
                ssrColor = totalColor.rgb / totalColor.a;
            } else {
                ssrColor = textureSampleLevel(colorMipChain, linearSamp, hitUV, roughness * roughness * compParams.maxMipLevel).rgb;
            }
        }
    }

    // ── Cubemap fallback ────────────────────────────────────────────────
    var envColor = vec3f(0.0);
    if (compParams.envCubemapEnabled > 0.5) {
        // Sample cubemap at roughness-dependent mip for glossy env reflections
        let cubeMipLevels = f32(textureNumLevels(envCubemap));
        let cubeMip = roughness * roughness * cubeMipLevels;
        envColor = textureSampleLevel(envCubemap, linearSamp, R, cubeMip).rgb;
    }

    // Blend SSR and cubemap based on confidence
    let envWeight  = compParams.envFallbackStr * (1.0 - confidence);
    let reflColor  = ssrColor * confidence + envColor * envWeight;
    let totalWeight = confidence + envWeight;

    // Apply Fresnel
    let ssrWeight = saturate(totalWeight);
    let finalColor = mix(sceneColor.rgb, reflColor, fresnel * ssrWeight);

    return vec4f(finalColor, sceneColor.a);
}
