// /src/shaders/deferred_lighting.wgsl
// Fullscreen deferred lighting pass.
// Reads G-Buffer textures + light storage buffer → HDR colour output.

// ---- Group 0: Per-frame uniforms ----------------------------------------

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

@group(0) @binding(0) var<uniform> frame : PerFrameUniforms;

// ---- Group 1: G-Buffer textures -----------------------------------------

@group(1) @binding(0) var gbAlbedoAO         : texture_2d<f32>;
@group(1) @binding(1) var gbNormalRoughness  : texture_2d<f32>;
@group(1) @binding(2) var gbMetallicEmissive : texture_2d<f32>;
@group(1) @binding(3) var gbDepth            : texture_depth_2d;
@group(1) @binding(4) var gbSampler          : sampler;

// ---- Group 2: Light storage + shadow atlas ------------------------------

struct Light {
    posRange       : vec4f,  // xyz = world position, w = range
    colorIntensity : vec4f,  // xyz = linear colour, w = intensity
    dirType        : vec4f,  // xyz = direction (directional/spot), w = type (0=dir,1=point,2=spot)
    coneAngles     : vec4f,  // x = cos(innerCone), y = cos(outerCone), z = shadowBias, w = castShadow
    shadowUV       : vec4f,  // xy = atlas UV offset, zw = atlas UV scale (cascade 0, legacy)
}

struct LightBuffer {
    count  : u32,
    _pad0  : u32,
    _pad1  : u32,
    _pad2  : u32,
    lights : array<Light>,
}

// Per-cascade shadow data — must match CPU layout in LightSystem.ts uploadShadowData().
struct CascadeInfo {
    viewProj : mat4x4f,  // light view-projection matrix            @ +0   (64 bytes)
    atlasUV  : vec4f,    // xy = UV offset in atlas, zw = UV scale  @ +64  (16 bytes)
    params   : vec4f,    // x=splitFar, y=ndcBias, z=projNear, w=projFar  @ +80 (16 bytes)
}

struct ShadowData {
    cascades : array<CascadeInfo, 6>,  // 6 × 96 = 576 bytes (covers CSM ≤4 and cube=6)
    params   : vec4f,                  // x=shadowType, y=numCascades, z=atlasTexelSize, w=pcfRadius @ +576
}

struct ShadowBuffer {
    lights : array<ShadowData, 256>,
}

@group(2) @binding(0) var<storage, read> lightBuf   : LightBuffer;
@group(2) @binding(1) var shadowAtlas                : texture_depth_2d;
@group(2) @binding(2) var shadowSampler              : sampler_comparison;
@group(2) @binding(3) var<storage, read> shadowBuf  : ShadowBuffer;

// ---- Group 3: Environment cubemap (IBL reflections) --------------------
//
// TODO: Replace with per-object reflection probes when probe system is
//       implemented.  For now the single scene-level environment cubemap
//       is used for all surfaces when present (controlled by envParams.enabled).

struct EnvParams {
    enabled  : u32,    // 1 = cubemap available, 0 = placeholder bound
    ambientR : f32,    // scene ambient light color (linear RGB)
    ambientG : f32,
    ambientB : f32,
}

@group(3) @binding(0) var<uniform> envParams : EnvParams;
@group(3) @binding(1) var          envCubemap : texture_cube<f32>;
@group(3) @binding(2) var          envSampler : sampler;

// ---- Vertex shader (fullscreen triangle) --------------------------------

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4f {
    // Full-screen triangle from three vertices, no vertex buffer needed.
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    return vec4f(pos[vi], 0.0, 1.0);
}

// ---- Helpers ------------------------------------------------------------

fn reconstructWorldPos(fragCoord: vec2f, depth: f32) -> vec3f {
    let res   = max(frame.resolution, vec2f(1.0));
    let uv    = fragCoord / res;
    let ndcX  = uv.x * 2.0 - 1.0;
    // Flip Y: clip Y+ = up, texture/fragCoord Y+ = down
    let ndcY  = (1.0 - uv.y) * 2.0 - 1.0;
    let clip  = vec4f(ndcX, ndcY, depth, 1.0);
    let world = frame.inverseViewProjection * clip;
    return world.xyz / world.w;
}

fn pointLightAttenuation(dist: f32, range: f32) -> f32 {
    if (range <= 0.0) { return 0.0; }
    let r = clamp(dist / range, 0.0, 1.0);
    // Smooth quadratic window that reaches zero at dist == range.
    return saturate(1.0 - r * r * r * r) / (dist * dist + 1.0);
}

fn distributionGGX(NdotH: f32, roughness: f32) -> f32 {
    let a    = roughness * roughness;
    let a2   = a * a;
    let d    = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d + 1e-7);
}

fn fresnelSchlick(VdotH: f32, F0: vec3f) -> vec3f {
    let t = clamp(1.0 - VdotH, 0.0, 1.0);
    let t2 = t * t;
    return F0 + (1.0 - F0) * (t2 * t2 * t);
}

// ---- Shadow helpers -----------------------------------------------------

// PCF sampling with configurable radius.
// texel: atlas texel size (1.0 / atlasSize), from sd.params.z.
// pcfRadius: number of taps in each direction (1=3×3, 2=5×5, 3=7×7), from sd.params.w.
// casc.params: x=splitFar, y=ndcBias, z=projNear, w=projFar.
// nDotL: dot(N, L) for slope-scale bias.
// Returns [0,1]: 1.0 = fully lit, 0.0 = fully in shadow.
fn sampleCascadeFromNDC(casc: CascadeInfo, ndc: vec3f, nDotL: f32, texel: f32, pcfR: i32) -> f32 {
    let rawBias = casc.params.y;
    let near    = casc.params.z;
    let far     = casc.params.w;
    let nDotLc  = clamp(nDotL, 0.25, 1.0);

    // For perspective projections (near > 0), a constant NDC bias maps to a
    // huge world-space offset at large depths due to the hyperbolic mapping.
    // We compute the depth-correct NDC bias at this sample's depth, but
    // enforce a minimum floor so that far faces still have enough bias to
    // prevent shadow acne (where there are no contact shadows anyway).
    var slopeBias: f32;
    if (near > 0.001) {
        let d = far - ndc.z * (far - near);
        let depthCorrect = rawBias * d * d / (near * far);
        // Floor: 10% of the constant NDC bias — enough to prevent acne at
        // far distances while keeping contacts tight (peter-panning ≤ ~1–2 cm).
        slopeBias = max(depthCorrect, rawBias * 0.1) / nDotLc;
    } else {
        // Ortho (CSM): linear depth mapping, rawBias is already correct.
        slopeBias = rawBias / nDotLc;
    }

    let shadowUV   = vec2f(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let atlasCoord = casc.atlasUV.xy + shadowUV * casc.atlasUV.zw;
    let depth      = ndc.z - slopeBias;

    // Clamp PCF taps to stay within this tile's atlas region.
    // Prevents bleeding into adjacent tiles which causes bright seams.
    let tileMin = casc.atlasUV.xy;
    let tileMax = casc.atlasUV.xy + casc.atlasUV.zw - texel;

    // NxN PCF kernel where N = 2*pcfR + 1.
    var shadow = 0.0;
    var count  = 0.0;
    for (var dx = -pcfR; dx <= pcfR; dx++) {
        for (var dy = -pcfR; dy <= pcfR; dy++) {
            let off = vec2f(f32(dx), f32(dy)) * texel;
            let tap = clamp(atlasCoord + off, tileMin, tileMax);
            shadow += textureSampleCompareLevel(shadowAtlas, shadowSampler, tap, depth);
            count += 1.0;
        }
    }
    return shadow / count;
}

fn sampleCascade(casc: CascadeInfo, worldPos: vec3f, nDotL: f32, texel: f32, pcfR: i32) -> f32 {
    let clipPos = casc.viewProj * vec4f(worldPos, 1.0);
    let ndc = clipPos.xyz / clipPos.w;

    if (ndc.x < -1.0 || ndc.x > 1.0 ||
        ndc.y < -1.0 || ndc.y > 1.0 ||
        ndc.z <  0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    return sampleCascadeFromNDC(casc, ndc, nDotL, texel, pcfR);
}

// Variant for cube-shadow face blending: clamps NDC x/y instead of rejecting.
fn sampleCascadeClamped(casc: CascadeInfo, worldPos: vec3f, nDotL: f32, texel: f32, pcfR: i32) -> f32 {
    let clipPos = casc.viewProj * vec4f(worldPos, 1.0);
    var ndc = clipPos.xyz / clipPos.w;

    if (ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    ndc.x = clamp(ndc.x, -1.0, 1.0);
    ndc.y = clamp(ndc.y, -1.0, 1.0);

    return sampleCascadeFromNDC(casc, ndc, nDotL, texel, pcfR);
}

// Evaluate shadow for light index `li` at world position `worldPos`.
// nDotL: dot(N, L) forwarded to sampleCascade for slope-scale bias.
// Returns 1.0 when fully lit (no shadow or no shadow data).
fn evalShadow(li: u32, worldPos: vec3f, nDotL: f32) -> f32 {
    let sd         = shadowBuf.lights[li];
    let shadowType = u32(sd.params.x + 0.5);
    let texel      = sd.params.z;                  // 1.0 / atlasSize
    let pcfR       = i32(sd.params.w + 0.5);       // PCF radius (1=3×3, 2=5×5, etc.)

    if (shadowType == 3u) {
        // Omnidirectional cube map (point light): select face by dominant axis.
        let lightPos = lightBuf.lights[li].posRange.xyz;
        let dir      = worldPos - lightPos;
        let absDir   = abs(dir);

        var face: u32;
        if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
            face = select(1u, 0u, dir.x >= 0.0);
        } else if (absDir.y >= absDir.z) {
            face = select(3u, 2u, dir.y >= 0.0);
        } else {
            face = select(5u, 4u, dir.z >= 0.0);
        }

        return sampleCascade(sd.cascades[face], worldPos, nDotL, texel, pcfR);
    } else if (shadowType == 2u) {
        // Cascaded shadow map: uses ortho projection, constant NDC bias is correct.
        let eyeDepth = -(frame.viewMatrix * vec4f(worldPos, 1.0)).z;
        let numCasc  = u32(sd.params.y + 0.5);
        var cascIdx  = numCasc - 1u;
        for (var ci = 0u; ci < 4u; ci++) {
            if (ci >= numCasc) { break; }
            if (eyeDepth < sd.cascades[ci].params.x) {
                cascIdx = ci;
                break;
            }
        }

        return sampleCascade(sd.cascades[cascIdx], worldPos, nDotL, texel, pcfR);
    } else if (shadowType == 1u) {
        // Standard single shadow map (spot light).
        return sampleCascade(sd.cascades[0], worldPos, nDotL, texel, pcfR);
    }
    return 1.0;
}

// ---- Fragment shader ----------------------------------------------------

@fragment
fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
    let res = max(frame.resolution, vec2f(1.0));
    let uv  = fragCoord.xy / res;

    // Sample G-Buffer colour attachments
    let albedoAO      = textureSample(gbAlbedoAO,         gbSampler, uv);
    let normalRough   = textureSample(gbNormalRoughness,  gbSampler, uv);
    let metalEmissive = textureSample(gbMetallicEmissive, gbSampler, uv);

    // Load depth without filtering (requires non-filtering sampler for depth).
    let texCoord = vec2i(i32(fragCoord.x), i32(fragCoord.y));
    let depth    = textureLoad(gbDepth, texCoord, 0);

    // Background: depth == 1.0 → no geometry written into this pixel.
    if (depth >= 1.0) {
        return vec4f(0.01, 0.01, 0.04, 1.0);   // very dark sky
    }

    // Guard: detect pixels where the depth prepass wrote a surface depth but
    // the G-Buffer fill pass failed the depth comparison (ULP rounding artefact
    // at silhouette edges).  Cleared normalRoughness.rgb is (0,0,0); any valid
    // encoded world-space normal produced by the G-Buffer pass lives in [0,1]³
    // and therefore has a non-zero squared length.  Treat these "ghost" pixels
    // as background to avoid the infinite-Gs specular spike they would produce.
    let rawNormal = normalRough.rgb;
    if (dot(rawNormal, rawNormal) < 1e-4) {
        return vec4f(0.01, 0.01, 0.04, 1.0);
    }

    // Decode G-Buffer
    let albedo    = albedoAO.rgb;
    let occlusion = albedoAO.a;
    let N         = normalize(normalRough.rgb * 2.0 - 1.0);  // [0,1] → [-1,1]
    let roughness = max(normalRough.a, 0.04);

    let metallic  = metalEmissive.r;

    // G-Buffer RT2: (encodedMetallic, emissive.r, emissive.g, emissive.b)
    let emissive  = metalEmissive.gba;

    let worldPos  = reconstructWorldPos(fragCoord.xy, depth);
    let V         = normalize(frame.cameraPosition - worldPos);

    // PBR reflectance at normal incidence
    let F0           = mix(vec3f(0.04), albedo, metallic);
    let diffuseColor = albedo * (1.0 - metallic);

    var Lo = vec3f(0.0);

    let lightCount = min(lightBuf.count, 256u);
    for (var i = 0u; i < lightCount; i++) {
        let light     = lightBuf.lights[i];
        let lightType = u32(light.dirType.w + 0.5); // round to nearest int

        var L        : vec3f;
        var radiance : vec3f;

        if (lightType == 0u) {
            // Directional light — dirType.xyz is pre-normalized on CPU; just negate for L.
            L        = -light.dirType.xyz;
            radiance = light.colorIntensity.rgb * light.colorIntensity.w;
        } else {
            // Positional light (point or spot) — common distance attenuation.
            let toLight = light.posRange.xyz - worldPos;
            let dist    = length(toLight);
            L           = toLight / max(dist, 1e-5);
            let atten   = pointLightAttenuation(dist, light.posRange.w);

            // Spotlight angular attenuation (type == 2).
            // coneAngles.x = cos(innerCone), coneAngles.y = cos(outerCone).
            // -L is the light→surface direction; light.dirType.xyz is the spotlight axis
            // (also pointing light→scene).  cosTheta ≈ 1 when the surface is on-axis.
            var spotAtten = 1.0;
            if (lightType == 2u) {
                let cosTheta = dot(-L, light.dirType.xyz);
                let innerCos = light.coneAngles.x;
                let outerCos = light.coneAngles.y;
                // Linear ramp in cosine space, then squared for a smoother penumbra edge.
                let t    = clamp((cosTheta - outerCos) / max(innerCos - outerCos, 1e-4), 0.0, 1.0);
                spotAtten = t * t;
            }

            radiance = light.colorIntensity.rgb * light.colorIntensity.w * atten * spotAtten;
        }

        let NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) { continue; }

        let H     = normalize(L + V);
        let NdotH = max(dot(N, H), 0.0);
        let VdotH = max(dot(V, H), 0.0);
        let NdotV = max(dot(N, V), 1e-4);

        // Cook-Torrance specular — height-correlated Smith G2 visibility.
        // Vis = G2 / (4·NdotL·NdotV).  Unlike the old 0.25/(NdotV·NdotL)
        // approximation, this is bounded at grazing angles and eliminates
        // the specular blowout at silhouette edges.
        let D   = distributionGGX(NdotH, roughness);
        let F   = fresnelSchlick(VdotH, F0);
        let a   = roughness * roughness;           // GGX alpha (matches distributionGGX)
        let a2  = a * a;
        let lV  = NdotL * sqrt(a2 + (1.0 - a2) * NdotV * NdotV);
        let lL  = NdotV * sqrt(a2 + (1.0 - a2) * NdotL * NdotL);
        let Vis = 0.5 / max(lV + lL, 1e-7);
        let specular = D * F * Vis;

        // Lambertian diffuse weighted by energy conservation
        let kD      = (vec3f(1.0) - F) * (1.0 - metallic);
        let diffuse = kD * diffuseColor * (1.0 / 3.14159265);

        // Shadow factor: 1.0 = fully lit, 0.0 = fully shadowed.
        var shadowFactor = 1.0f;
        if (light.coneAngles.w > 0.5) {
            shadowFactor = evalShadow(i, worldPos, NdotL);
        }

        Lo += (diffuse + specular) * radiance * NdotL * shadowFactor;
    }

    // ---- Environment cubemap IBL specular reflection ----------------------
    //
    // TODO: When reflection probes are implemented, select the probe cubemap
    //       closest to worldPos instead of the global environment cubemap.
    //       The probe should supply both a prefiltered specular map and an
    //       irradiance map for diffuse IBL.
    //
    // For now: single scene-level cubemap, no prefiltering, no split-sum BRDF
    // LUT.  Roughness attenuates the reflection analytically.

    var envIBL = vec3f(0.0);
    if (envParams.enabled == 1u) {
        let R          = reflect(-V, N);
        let envSample  = textureSampleLevel(envCubemap, envSampler, R, 0.0).rgb;
        let F_env      = fresnelSchlick(max(dot(N, V), 0.0), F0);
        // Rough surfaces reflect less — squared falloff approximates the
        // energy loss from not using a prefiltered cubemap + split-sum BRDF.
        let smoothness = (1.0 - roughness) * (1.0 - roughness);
        envIBL         = envSample * F_env * smoothness;
    }

    // Ambient (scene-configurable) + emissive + IBL
    let ambient = vec3f(envParams.ambientR, envParams.ambientG, envParams.ambientB) * albedo * occlusion;
    let color   = ambient + Lo + emissive + envIBL;

    return vec4f(color, 1.0);
}
