// /src/shaders/forward_transparent.wgsl
//
// Forward PBR lighting pass for alpha-blended (transparent) materials.
// Runs after deferred lighting + post-process, reading the already-lit scene
// as the blend destination.
//
// Bind groups:
//   @group(0) @binding(0)  PerFrameUniforms
//   @group(1) @binding(0)  MaterialUniforms (PBR params incl. opacity + IOR)
//   @group(1) @binding(1)  materialSampler   (HAS_TEXTURES)
//   @group(1) @binding(2)  baseColorMap       (HAS_BASE_COLOR_MAP)
//   @group(1) @binding(3)  normalMap          (HAS_NORMAL_MAP)
//   @group(1) @binding(4)  mrMap              (HAS_MR_MAP)
//   @group(1) @binding(5)  aoMap              (HAS_AO_MAP)
//   @group(1) @binding(6)  emissiveMap        (HAS_EMISSIVE_MAP)
//   @group(2) @binding(0)  ModelUniforms (per-draw model matrix)
//   @group(3) @binding(0)  LightBuffer (storage)
//   @group(3) @binding(1)  shadowAtlas (depth texture)
//   @group(3) @binding(2)  shadowSampler (comparison sampler)
//   @group(3) @binding(3)  ShadowBuffer (storage)
//   @group(3) @binding(4)  EnvParams (uniform)
//   @group(3) @binding(5)  envCubemap (texture_cube)
//   @group(3) @binding(6)  envSampler (sampler)

// ── Group 0: Per-frame uniforms ──────────────────────────────────────────

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

// Scene color copy for screen-space refraction (snapshot of the lit scene
// taken before the transparent pass).
@group(0) @binding(1) var sceneColor        : texture_2d<f32>;
@group(0) @binding(2) var sceneColorSampler : sampler;

// ── Group 1: Material uniforms + textures ────────────────────────────────

struct MaterialUniforms {
    baseColorFactor   : vec4f,
    metallicFactor    : f32,
    roughnessFactor   : f32,
    normalScale       : f32,
    occlusionStrength : f32,
    emissiveFactor    : vec3f,
    alphaCutoff       : f32,
    opacity           : f32,
    ior               : f32,
    _pad0             : f32,
    _pad1             : f32,
}

@group(1) @binding(0) var<uniform> material : MaterialUniforms;

#ifdef HAS_TEXTURES
@group(1) @binding(1) var materialSampler : sampler;
#endif

#ifdef HAS_BASE_COLOR_MAP
@group(1) @binding(2) var baseColorMap : texture_2d<f32>;
#endif

#ifdef HAS_NORMAL_MAP
@group(1) @binding(3) var normalMap : texture_2d<f32>;
#endif

#ifdef HAS_MR_MAP
@group(1) @binding(4) var mrMap : texture_2d<f32>;
#endif

#ifdef HAS_AO_MAP
@group(1) @binding(5) var aoMap : texture_2d<f32>;
#endif

#ifdef HAS_EMISSIVE_MAP
@group(1) @binding(6) var emissiveMap : texture_2d<f32>;
#endif

// ── Group 2: Per-draw model matrix ───────────────────────────────────────

struct ModelUniforms {
    modelMatrix : mat4x4f,
}

@group(2) @binding(0) var<uniform> draw : ModelUniforms;

// ── Group 3: Lights + shadows + environment ──────────────────────────────

struct Light {
    posRange       : vec4f,
    colorIntensity : vec4f,
    dirType        : vec4f,
    coneAngles     : vec4f,
    shadowUV       : vec4f,
}

struct LightBuffer {
    count  : u32,
    _pad0  : u32,
    _pad1  : u32,
    _pad2  : u32,
    lights : array<Light>,
}

struct CascadeInfo {
    viewProj : mat4x4f,
    atlasUV  : vec4f,
    params   : vec4f,
}

struct ShadowData {
    cascades : array<CascadeInfo, 6>,
    params   : vec4f,
}

struct ShadowBuffer {
    lights : array<ShadowData, 256>,
}

struct EnvParams {
    enabled  : u32,
    ambientR : f32,
    ambientG : f32,
    ambientB : f32,
}

@group(3) @binding(0) var<storage, read> lightBuf    : LightBuffer;
@group(3) @binding(1) var shadowAtlas                 : texture_depth_2d;
@group(3) @binding(2) var shadowSampler               : sampler_comparison;
@group(3) @binding(3) var<storage, read> shadowBuf   : ShadowBuffer;
@group(3) @binding(4) var<uniform>       envParams   : EnvParams;
@group(3) @binding(5) var                envCubemap   : texture_cube<f32>;
@group(3) @binding(6) var                envSampler   : sampler;

// ── Vertex I/O ───────────────────────────────────────────────────────────

struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) tangent  : vec4f,
    @location(3) uv0      : vec2f,
}

struct VertexOutput {
    @builtin(position) clipPos   : vec4f,
    @location(0)       worldPos  : vec3f,
    @location(1)       worldNorm : vec3f,
    @location(2)       uv0       : vec2f,
    @location(3)       worldTan  : vec3f,
    @location(4)       tanSign   : f32,
}

// ── Vertex shader ────────────────────────────────────────────────────────

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let worldPos  = (draw.modelMatrix * vec4f(in.position, 1.0)).xyz;
    let worldNorm = normalize((draw.modelMatrix * vec4f(in.normal, 0.0)).xyz);
    let worldTan  = normalize((draw.modelMatrix * vec4f(in.tangent.xyz, 0.0)).xyz);

    var out: VertexOutput;
    out.clipPos   = frame.viewProjectionMatrix * vec4f(worldPos, 1.0);
    out.worldPos  = worldPos;
    out.worldNorm = worldNorm;
    out.uv0       = in.uv0;
    out.worldTan  = worldTan;
    out.tanSign   = in.tangent.w;
    return out;
}

// ── PBR Helpers ──────────────────────────────────────────────────────────

fn distributionGGX(NdotH: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom + 1e-7);
}

fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
    let t = 1.0 - cosTheta;
    let t2 = t * t;
    return F0 + (1.0 - F0) * (t2 * t2 * t);
}

fn fresnelSchlickIOR(cosTheta: f32, ior: f32) -> f32 {
    let f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
    let t  = 1.0 - cosTheta;
    let t2 = t * t;
    return f0 + (1.0 - f0) * (t2 * t2 * t);
}

fn pointLightAttenuation(dist: f32, range: f32) -> f32 {
    if (range <= 0.0) { return 0.0; }
    let r = clamp(dist / range, 0.0, 1.0);
    return saturate(1.0 - r * r * r * r) / (dist * dist + 1.0);
}

// ── Shadow evaluation (simplified — no CSM cascade blend for forward) ────

fn sampleShadowPCF(casc: CascadeInfo, ndc: vec3f, texel: f32, pcfR: i32) -> f32 {
    let atlasCoord = casc.atlasUV.xy + ndc.xy * casc.atlasUV.zw;
    let tileMin    = casc.atlasUV.xy;
    let tileMax    = casc.atlasUV.xy + casc.atlasUV.zw;

    let rawBias  = casc.params.y;
    let depth    = ndc.z - rawBias;

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

fn evalShadow(lightIdx: u32, worldPos: vec3f, NdotL: f32) -> f32 {
    let sd         = shadowBuf.lights[lightIdx];
    let shadowType = u32(sd.params.x + 0.5);
    let texel      = sd.params.z;
    let pcfR       = i32(sd.params.w + 0.5);

    if (shadowType == 0u) { return 1.0; }

    // Standard (spot) or first cascade (CSM) or dominant cube face
    var cascIdx = 0u;

    if (shadowType == 2u) {
        // CSM: find cascade by eye depth
        let eyeDepth = -(frame.viewMatrix * vec4f(worldPos, 1.0)).z;
        let numCasc = u32(sd.params.y + 0.5);
        cascIdx = numCasc - 1u;
        for (var ci = 0u; ci < 4u; ci++) {
            if (ci >= numCasc) { break; }
            if (eyeDepth < sd.cascades[ci].params.x) {
                cascIdx = ci;
                break;
            }
        }
    } else if (shadowType == 3u) {
        // Cube: select face by dominant axis
        let lp = lightBuf.lights[lightIdx].posRange.xyz;
        let d  = worldPos - lp;
        let ad = abs(d);
        if (ad.x >= ad.y && ad.x >= ad.z) {
            cascIdx = select(1u, 0u, d.x > 0.0);
        } else if (ad.y >= ad.z) {
            cascIdx = select(3u, 2u, d.y > 0.0);
        } else {
            cascIdx = select(5u, 4u, d.z > 0.0);
        }
    }

    let casc = sd.cascades[cascIdx];
    let lsPos = casc.viewProj * vec4f(worldPos, 1.0);
    let ndc = vec3f(lsPos.xy / lsPos.w * 0.5 + 0.5, lsPos.z / lsPos.w);

    if (ndc.x < 0.0 || ndc.x > 1.0 || ndc.y < 0.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    return sampleShadowPCF(casc, ndc, texel, pcfR);
}

// ── Fragment shader ──────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) frontFacing: bool) -> @location(0) vec4f {
    // ── Base color ───────────────────────────────────────────────────────
    var baseColor = material.baseColorFactor;
#ifdef HAS_BASE_COLOR_MAP
    baseColor *= textureSample(baseColorMap, materialSampler, in.uv0);
#endif
    let albedo = baseColor.rgb;
    let alpha  = baseColor.a * material.opacity;

    if (alpha < 0.001) { discard; }

    // ── Normal ──────────────────────────────────────────────────────────
    var N = normalize(in.worldNorm);
#ifdef DOUBLE_SIDED
    if (!frontFacing) { N = -N; }
#endif

#ifdef HAS_NORMAL_MAP
    let nm  = textureSample(normalMap, materialSampler, in.uv0).rgb * 2.0 - 1.0;
    let T   = normalize(in.worldTan);
    let B   = cross(N, T) * in.tanSign;
    let TBN = mat3x3f(T, B, N);
    N = normalize(TBN * (nm * vec3f(material.normalScale, material.normalScale, 1.0)));
#endif

    // ── Metallic / Roughness ────────────────────────────────────────────
    var metallic  = material.metallicFactor;
    var roughness = material.roughnessFactor;
#ifdef HAS_MR_MAP
    let mrSample = textureSample(mrMap, materialSampler, in.uv0);
    roughness *= mrSample.g;
    metallic  *= mrSample.b;
#endif
    roughness = clamp(roughness, 0.04, 1.0);

    // ── Occlusion ───────────────────────────────────────────────────────
    var occlusion = 1.0;
#ifdef HAS_AO_MAP
    occlusion = mix(1.0, textureSample(aoMap, materialSampler, in.uv0).r, material.occlusionStrength);
#endif

    // ── Emissive ────────────────────────────────────────────────────────
    var emissive = material.emissiveFactor;
#ifdef HAS_EMISSIVE_MAP
    emissive *= textureSample(emissiveMap, materialSampler, in.uv0).rgb;
#endif

    // ── PBR Setup ───────────────────────────────────────────────────────
    let V  = normalize(frame.cameraPosition - in.worldPos);
    let NdotV = max(dot(N, V), 1e-4);

    // IOR-derived F0: specular reflectance at normal incidence.
    // For dielectrics this replaces the fixed 0.04; for metals F0 = albedo.
    let dielectricF0 = pow((material.ior - 1.0) / (material.ior + 1.0), 2.0);
    let F0 = mix(vec3f(dielectricF0), albedo, metallic);
    let diffuseColor = albedo * (1.0 - metallic);

    // ── IOR-based Fresnel for transparency ──────────────────────────────
    // At grazing angles the surface becomes fully reflective (opaque).
    // At head-on view the surface is at material opacity.
    let fresnelTransparency = fresnelSchlickIOR(NdotV, material.ior);

    // ── Light loop ──────────────────────────────────────────────────────
    var Lo = vec3f(0.0);
    let lightCount = min(lightBuf.count, 256u);

    for (var i = 0u; i < lightCount; i++) {
        let light     = lightBuf.lights[i];
        let lightType = u32(light.dirType.w + 0.5);

        var L        : vec3f;
        var radiance : vec3f;

        if (lightType == 0u) {
            L        = -light.dirType.xyz;
            radiance = light.colorIntensity.rgb * light.colorIntensity.w;
        } else {
            let toLight = light.posRange.xyz - in.worldPos;
            let dist    = length(toLight);
            L           = toLight / max(dist, 1e-5);
            let atten   = pointLightAttenuation(dist, light.posRange.w);

            var spotAtten = 1.0;
            if (lightType == 2u) {
                let cosTheta = dot(-L, light.dirType.xyz);
                let innerCos = light.coneAngles.x;
                let outerCos = light.coneAngles.y;
                let t = clamp((cosTheta - outerCos) / max(innerCos - outerCos, 1e-4), 0.0, 1.0);
                spotAtten = t * t;
            }

            radiance = light.colorIntensity.rgb * light.colorIntensity.w * atten * spotAtten;
        }

        let NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) { continue; }

        let H     = normalize(L + V);
        let NdotH = max(dot(N, H), 0.0);
        let VdotH = max(dot(V, H), 0.0);

        let D   = distributionGGX(NdotH, roughness);
        let F   = fresnelSchlick(VdotH, F0);
        let a   = roughness * roughness;
        let a2  = a * a;
        let lV  = NdotL * sqrt(a2 + (1.0 - a2) * NdotV * NdotV);
        let lL  = NdotV * sqrt(a2 + (1.0 - a2) * NdotL * NdotL);
        let Vis = 0.5 / max(lV + lL, 1e-7);
        let specular = D * F * Vis;

        let kD      = (vec3f(1.0) - F) * (1.0 - metallic);
        let diffuse = kD * diffuseColor * (1.0 / 3.14159265);

        var shadowFactor = 1.0f;
        if (light.coneAngles.w > 0.5) {
            shadowFactor = evalShadow(i, in.worldPos, NdotL);
        }

        Lo += (diffuse + specular) * radiance * NdotL * shadowFactor;
    }

    // ── Environment IBL ─────────────────────────────────────────────────
    var envIBL = vec3f(0.0);
    if (envParams.enabled == 1u) {
        let R         = reflect(-V, N);
        let envSample = textureSampleLevel(envCubemap, envSampler, R, 0.0).rgb;
        let F_env     = fresnelSchlick(NdotV, F0);
        let smoothness = (1.0 - roughness) * (1.0 - roughness);
        envIBL        = envSample * F_env * smoothness;
    }

    // ── Ambient + emissive ──────────────────────────────────────────────
    let ambient = vec3f(envParams.ambientR, envParams.ambientG, envParams.ambientB) * albedo * occlusion;
    let color   = ambient + Lo + emissive + envIBL;

    // ── Screen-space refraction (fake) ──────────────────────────────────
    //
    // Use the angle between the camera view direction and the surface normal
    // to offset where we sample the background. Normals facing the camera
    // (small angle) → sample directly behind the pixel. Normals at larger
    // angles → shift the sample position outward along the screen-space normal.
    let screenUV = in.clipPos.xy / frame.resolution;

    // The deviation angle: 0 when normal faces camera, grows toward edges.
    let angle = acos(clamp(NdotV, 0.0, 1.0));

    // IOR controls how strong the offset is. ior=1 → no distortion.
    let iorStrength = (material.ior - 1.0) * 0.5;

    // Project the world normal into screen space for offset direction.
    let viewN = normalize((frame.viewMatrix * vec4f(N, 0.0)).xyz);

    // Offset magnitude scales with the angle — bigger angle, more shift.
    let offset = viewN.xy * angle * iorStrength;

    let refractedUV = clamp(screenUV + offset, vec2f(0.001), vec2f(0.999));
    let refractedScene = textureSampleLevel(sceneColor, sceneColorSampler, refractedUV, 0.0).rgb;

    // ── Transparency compositing ─────────────────────────────────────────
    //
    // The final pixel blends three contributions:
    //   1. Refracted background — seen through the transparent surface
    //   2. Surface lighting     — the object's own PBR shading (diffuse + specular)
    //   3. Env reflections      — always on top (reflected light)
    //
    // Fresnel controls the reflected/transmitted split:
    //   At grazing angles → mostly reflection (opaque, no background visible)
    //   At normal incidence → mostly transmission (refracted background visible)
    let surfaceLighting = ambient + Lo + emissive;
    let transmission    = (1.0 - fresnelTransparency) * (1.0 - alpha * 0.5);

    // Blend: refracted background * transmission + surface lighting * opacity + reflections
    let finalColor = refractedScene * transmission
                   + surfaceLighting * alpha * (1.0 - fresnelTransparency)
                   + envIBL * fresnelTransparency;

    // Output with alpha = 1 since we're doing our own compositing with the
    // refracted background sample (no need for hardware alpha blending to
    // show the scene behind — we already sampled and mixed it in).
    return vec4f(finalColor, 1.0);
}
