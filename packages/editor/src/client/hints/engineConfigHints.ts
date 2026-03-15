import type { HintContent } from '../composables/useHintPanel';

type HintDef = Omit<HintContent, 'anchorY'>;

export const engineConfigHints: Record<string, HintDef> = {

    // ── Rendering ────────────────────────────────────────────────────────

    frustumCulling: {
        title: 'Frustum Culling',
        body: `
<p>When enabled, objects whose bounding sphere lies entirely outside
the camera frustum are <strong>skipped</strong> during rendering.</p>
<p>Each mesh's AABB is converted to a bounding sphere, transformed to
world space, and tested against the 6 frustum planes. Objects that
fail the test are not submitted for drawing.</p>
<p class="hint-note">Disabling culling forces every mesh to be drawn every frame,
useful for debugging but very expensive in large scenes.</p>`,
    },
    frustumCullTolerance: {
        title: 'Cull Tolerance',
        body: `
<p>A <strong>world-space buffer</strong> (in units) added to each bounding
sphere's radius before frustum testing.</p>
<p><code>effectiveRadius = sphereRadius &times; maxScale + tolerance</code></p>
<div class="hint-values">
<div><code>0.0</code> — tight culling, objects pop at screen edges</div>
<div><code>2.0</code> — default, keeps objects slightly outside the view</div>
<div><code>5.0+</code> — very conservative, good for large animated meshes</div>
</div>
<p class="hint-note">Higher tolerance prevents popping for animated meshes whose
bounding sphere may not perfectly enclose all poses, and keeps
shadow-casting geometry alive beyond the screen edge.</p>`,
    },
    tonemap: {
        title: 'Tonemap Operator',
        body: `
<p>The <strong>HDR &rarr; LDR</strong> mapping applied in the final blit pass,
after exposure scaling.</p>
<p>Pipeline: <code>ldr = tonemap(hdrColor &times; exposure)</code></p>
<div class="hint-values">
<div><strong>ACES</strong> — Narkowicz 2015 approximation. Good contrast,
slight warm shift. Industry standard for film-like look.</div>
<div><strong>AgX</strong> — Blender's default. Better highlight handling,
less saturated clipping.</div>
<div><strong>Reinhard</strong> — Simple <code>x / (1 + x)</code>. Softer rolloff,
can look washed out.</div>
<div><strong>PBR Neutral</strong> — Khronos neutral tonemap. Minimal color
shift, designed for material accuracy.</div>
</div>
<p class="hint-note">Currently only <strong>ACES</strong> is fully implemented in the
shader. Other operators are declared in the config interface
for future implementation.</p>`,
    },

    // ── Ambient ──────────────────────────────────────────────────────────

    ambient: {
        title: 'Ambient Light (RGB)',
        body: `
<p>A constant <strong>scene-wide base illumination</strong> applied to all
surfaces in the deferred lighting pass, in linear RGB.</p>
<p>Applied as: <code>ambient = ambientColor &times; albedo &times; occlusion</code></p>
<p>Added to the final color alongside direct lighting, emissive, and
environment reflections.</p>
<div class="hint-values">
<div><code>[0, 0, 0]</code> — no ambient (fully dark shadows)</div>
<div><code>[0.03, 0.03, 0.03]</code> — default (subtle fill)</div>
<div><code>[0.1, 0.1, 0.1]</code> — noticeable fill, softens shadows</div>
</div>
<p class="hint-note">This is a flat constant — not image-based lighting (IBL).
For cubemap-based ambient, enable Env Reflections on materials
and load an environment cubemap.</p>`,
    },

    // ── Background ───────────────────────────────────────────────────────

    bgType: {
        title: 'Background Type',
        body: `
<p>How the sky/background is rendered for pixels not covered by geometry.
Uses <code>depthCompare: 'equal'</code> at depth 1.0 to only fill empty pixels.</p>
<div class="hint-values">
<div><strong>Color</strong> — solid flat color</div>
<div><strong>Gradient</strong> — vertical blend from top color to bottom color</div>
<div><strong>Texture</strong> — 2D image mapped to screen UV</div>
<div><strong>Cubemap</strong> — environment cube map sampled by view ray direction
(reconstructed from inverse view-projection matrix)</div>
</div>`,
    },
    bgColor: {
        title: 'Background Color',
        body: `
<p>The solid background color in <strong>linear RGB</strong>, used when
Background Type is set to <code>Color</code>.</p>
<p>Rendered as a full-screen quad behind all geometry.</p>`,
    },
    bgGradientTop: {
        title: 'Gradient Top Color',
        body: `
<p>The color at the <strong>top of the screen</strong> for gradient backgrounds,
in linear RGB. Blends smoothly down to the bottom color.</p>`,
    },
    bgGradientBottom: {
        title: 'Gradient Bottom Color',
        body: `
<p>The color at the <strong>bottom of the screen</strong> for gradient
backgrounds, in linear RGB.</p>`,
    },
    bgTexture: {
        title: 'Background Texture',
        body: `
<p>A 2D image displayed as the scene background. Sampled using screen
UV coordinates (stretched to fill the viewport).</p>
<p class="hint-note">For panoramic environments, use <strong>Cubemap</strong> type
instead — it correctly handles camera rotation.</p>`,
    },
    bgCubemap: {
        title: 'Background Cubemap',
        body: `
<p>A 6-face environment cubemap displayed as the sky. The view ray
direction is reconstructed per-pixel from the inverse view-projection
matrix and used to sample the cubemap.</p>
<p>This is also the cubemap used for <strong>environment reflections</strong>
on materials with Env Reflections enabled.</p>
<p class="hint-note">Cubemaps are loaded from 6 face images:
<code>posx, negx, posy, negy, posz, negz</code>.</p>`,
    },

    // ── SSAO ─────────────────────────────────────────────────────────────

    ssaoEnabled: {
        title: 'SSAO — Enabled',
        body: `
<p><strong>Screen-Space Ambient Occlusion</strong> darkens crevices, corners,
and contact areas where indirect light would be blocked in reality.</p>
<p>Runs as a 3-pass post-process at <strong>half resolution</strong>:</p>
<div class="hint-values">
<div>1. <strong>Compute</strong> — hemisphere sampling around G-Buffer normals</div>
<div>2. <strong>Bilateral blur</strong> — 5&times;5 depth-aware smoothing</div>
<div>3. <strong>Composite</strong> — bilinear upscale and multiply onto scene</div>
</div>
<p class="hint-note">SSAO uses temporal accumulation with history clamping
(3&times;3 neighbourhood min/max) and velocity-based disocclusion
to reduce noise without ghosting.</p>`,
    },
    ssaoRadius: {
        title: 'SSAO — Radius',
        body: `
<p><strong>World-space radius</strong> of the hemisphere used for sampling
around each pixel's surface point.</p>
<div class="hint-values">
<div><code>0.1–0.3</code> — fine detail AO (small crevices)</div>
<div><code>0.5</code> — default, good general-purpose</div>
<div><code>1.0–3.0</code> — large-scale contact shadows</div>
</div>
<p>Larger radius catches broader occlusion but requires more samples
to avoid banding. The samples are positioned within this radius
in view space and tested against the depth buffer.</p>`,
    },
    ssaoBias: {
        title: 'SSAO — Bias',
        body: `
<p>Depth bias to prevent <strong>self-occlusion</strong> on flat surfaces.
Prevents the surface from occluding itself due to depth buffer
precision limits.</p>
<p>Applied as a smooth ramp:
<code>smoothstep(&minus;bias&times;2, &minus;bias&times;0.5, depthDiff)</code></p>
<div class="hint-values">
<div><code>0.01</code> — tight, may show noise on flat surfaces</div>
<div><code>0.02</code> — default</div>
<div><code>0.05</code> — aggressive, reduces AO detail in shallow crevices</div>
</div>`,
    },
    ssaoIntensity: {
        title: 'SSAO — Intensity',
        body: `
<p><strong>Strength multiplier</strong> for the final AO darkening effect.</p>
<p>Applied as: <code>ao = 1.0 &minus; (rawAO / sampleCount) &times; intensity</code></p>
<div class="hint-values">
<div><code>0.5</code> — subtle, barely noticeable</div>
<div><code>1.5</code> — default, clear occlusion in crevices</div>
<div><code>3.0+</code> — dramatic darkening, stylized look</div>
</div>`,
    },
    ssaoSamples: {
        title: 'SSAO — Sample Count',
        body: `
<p>Number of <strong>hemisphere samples per pixel</strong> used to estimate
ambient occlusion. More samples = less noise, more GPU cost.</p>
<div class="hint-values">
<div><code>4</code> — very noisy, needs heavy blur</div>
<div><code>16</code> — default, good balance with temporal accumulation</div>
<div><code>32</code> — high quality, minimal noise</div>
</div>
<p class="hint-note">Temporal accumulation blends across frames, so even 16 samples
produce smooth results after a few frames. Higher counts mainly
help on first-frame quality and during fast camera motion.</p>`,
    },
    ssaoBlurSharpness: {
        title: 'SSAO — Blur Sharpness',
        body: `
<p>Controls the <strong>depth-aware edge preservation</strong> of the bilateral
blur pass. Higher values preserve more depth edges, preventing AO
from bleeding across object boundaries.</p>
<p>The blur computes a weight:
<code>w = exp(&minus;(centerDepth &minus; sampleDepth) &times; sharpness)</code></p>
<div class="hint-values">
<div><code>1–5</code> — soft blur, AO may bleed across edges</div>
<div><code>10</code> — default, good edge preservation</div>
<div><code>20–50</code> — very sharp edges, minimal bleed</div>
</div>`,
    },

    // ── Init-Time Config ─────────────────────────────────────────────────

    shadowAtlas: {
        title: 'Shadow Atlas Size',
        body: `
<p>Resolution of the shared <strong>shadow atlas texture</strong> (depth32float).
All shadow maps from all lights are packed into this single texture
using a strip allocator.</p>
<p>Set at engine init — cannot be changed at runtime.</p>
<div class="hint-values">
<div><code>2048</code> — small, fits few lights or low-res shadows</div>
<div><code>4096</code> — default, good for most scenes</div>
<div><code>8192</code> — large, allows many high-res shadow maps</div>
</div>`,
    },
    defaultShadowRes: {
        title: 'Default Shadow Resolution',
        body: `
<p>Per-cascade/face shadow map size used for lights that don't
specify their own <code>shadowMapResolution</code>.</p>
<p>Set at engine init.</p>`,
    },
    defaultCascades: {
        title: 'Default CSM Cascades',
        body: `
<p>Default number of cascade slices for directional lights that don't
specify <code>numCascades</code>. Range: 1&ndash;4.</p>`,
    },
    defaultBias: {
        title: 'Default Shadow Bias',
        body: `
<p>Default shadow bias for lights that don't specify their own.
Interpretation depends on shadow type (texel multiplier for CSM,
world-space offset for spot/cube).</p>`,
    },
    maxLights: {
        title: 'Max Lights',
        body: `
<p>Maximum number of lights supported in the scene. Lights beyond
this limit are ignored. Set at engine init.</p>`,
    },
    lightsPerCluster: {
        title: 'Lights per Cluster',
        body: `
<p>Maximum lights per cluster bin in the <strong>clustered lighting</strong>
structure. The view frustum is divided into a 3D grid; each bin
stores up to this many affecting lights.</p>
<p class="hint-note">If a cluster exceeds this limit, excess lights are silently
dropped for that cluster.</p>`,
    },
    maxAnisotropy: {
        title: 'Max Anisotropy',
        body: `
<p>Maximum <strong>anisotropic filtering</strong> level for texture samplers.
Improves texture clarity at oblique viewing angles.</p>
<div class="hint-values">
<div><code>1</code> — disabled (isotropic only)</div>
<div><code>4</code> — good balance of quality/performance</div>
<div><code>16</code> — maximum quality</div>
</div>`,
    },
    maxDraws: {
        title: 'Max Draws per Frame',
        body: `
<p>Maximum non-instanced draw calls per frame. Controls the size of
the model-matrix uniform buffer (<code>maxDraws &times; 256 bytes</code>)
and pre-allocated bind group arrays.</p>
<p>Objects beyond this limit cannot be drawn.</p>`,
    },
    maxInstances: {
        title: 'Max Static Instances',
        body: `
<p>Maximum number of static (instanced) objects that can be batched
per frame. Controls the size of the instance storage buffer
(<code>maxInstances &times; 64 bytes</code>).</p>
<p>Static objects sharing the same mesh + material are automatically
merged into single GPU draw calls with <code>instanceCount &gt; 1</code>.</p>`,
    },

    halfResolution: {
        title: 'Half Resolution',
        body: `
<p>When enabled, the engine renders at <strong>half the native pixel
resolution</strong> and the canvas is stretched back to full size via CSS.</p>
<p>This is a simple performance hack that reduces the pixel count by
~4&times;, trading visual sharpness for significantly lower GPU fill-rate
cost. Useful on integrated GPUs or high-DPI displays.</p>
<p class="hint-note">Requires a scene reload to take effect (init-time setting).</p>`,
    },
};
