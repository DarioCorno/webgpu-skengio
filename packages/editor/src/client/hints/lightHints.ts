import type { HintContent } from '../composables/useHintPanel';

type HintDef = Omit<HintContent, 'anchorY'>;

export const lightHints: Record<string, HintDef> = {
    type: {
        title: 'Light Type',
        body: `
<p>Determines how the light emits radiance into the scene.</p>
<div class="hint-values">
<div><strong>Directional</strong> — parallel rays from infinite distance (sun/moon).
No position, only direction. Uses CSM shadows.</div>
<div><strong>Point</strong> — emits in all directions from a position.
Has a range for distance attenuation. Uses 6-face cube shadow maps.</div>
<div><strong>Spot</strong> — emits in a cone from a position along a direction.
Has range + inner/outer cone angles. Uses a single standard shadow map.</div>
</div>`,
    },
    color: {
        title: 'Light Color (RGB)',
        body: `
<p>The color of the emitted light in <strong>linear sRGB</strong>.</p>
<p>Packed with intensity into a single <code>vec4f</code> on the GPU:
<code>colorIntensity = vec4(color.rgb, intensity)</code></p>
<p>The final radiance contribution is:</p>
<p><code>radiance = color &times; intensity &times; attenuation</code></p>
<p class="hint-note">Values are in [0, 1] per channel. To create a warm light,
use higher R values (e.g. <code>[1.0, 0.85, 0.6]</code>).
For cool light, boost B (e.g. <code>[0.6, 0.8, 1.0]</code>).</p>`,
    },
    intensity: {
        title: 'Intensity',
        body: `
<p>A dimensionless <strong>brightness multiplier</strong> applied to the light color.</p>
<p>The shader computes radiance as:</p>
<p><code>radiance = color &times; intensity &times; attenuation &times; spotAtten</code></p>
<div class="hint-values">
<div><code>1.0</code> — subtle fill light</div>
<div><code>5–15</code> — typical indoor point/spot light</div>
<div><code>20–50</code> — bright outdoor or dramatic light</div>
</div>
<p class="hint-note">For directional lights (sun), intensity is not attenuated by
distance — only <code>color &times; intensity</code> is used directly.</p>`,
    },
    range: {
        title: 'Range',
        body: `
<p>Maximum reach of the light in <strong>world units</strong>. Beyond this distance,
the light contribution drops to exactly zero.</p>
<p>The attenuation formula uses an inverse-square law with a smooth
window function:</p>
<p><code>atten = saturate(1 &minus; (d/range)&sup4;) / (d&sup2; + 1)</code></p>
<div class="hint-values">
<div><code>1 &minus; r&sup4;</code> — smooth falloff that reaches zero at <code>d = range</code></div>
<div><code>1 / (d&sup2; + 1)</code> — physically-based inverse-square with epsilon</div>
</div>
<p>Smaller range = tighter pool of light and cheaper GPU cost (fewer
affected cluster bins in the clustered lighting pass).</p>
<p class="hint-note">For shadow-casting lights, the range also defines the shadow
camera's far plane. A large range can reduce shadow depth precision.</p>`,
    },
    innerCone: {
        title: 'Inner Cone Angle',
        body: `
<p>The half-angle (in radians) of the spotlight's <strong>fully-lit inner cone</strong>.
Inside this cone, spot attenuation is 1.0 (full brightness).</p>
<p>Converted to cosine on the CPU before GPU upload:
<code>cos(innerConeAngle)</code></p>
<p>The spotlight falloff interpolates between inner and outer cones
in cosine space, then squares the result for a smooth penumbra:</p>
<p><code>t = clamp((cos&theta; &minus; cosOuter) / (cosInner &minus; cosOuter))</code><br/>
<code>spotAtten = t &times; t</code></p>
<div class="hint-values">
<div><code>0.0</code> — point source, no fully-lit core</div>
<div><code>0.3–0.5</code> — typical spotlight with visible hotspot</div>
</div>`,
    },
    outerCone: {
        title: 'Outer Cone Angle',
        body: `
<p>The half-angle (in radians) of the spotlight's <strong>outer boundary</strong>.
Outside this cone, the light contribution is zero.</p>
<p>The region between inner and outer cone angles is the
<strong>penumbra</strong> — a smooth falloff zone.</p>
<div class="hint-values">
<div><code>0.5–0.8</code> — typical spotlight, soft edges</div>
<div><code>1.57</code> (90&deg;) — hemisphere, maximum spread</div>
</div>
<p class="hint-note">The outer angle must be &ge; inner angle. If they are equal,
there is no penumbra and the edge is a hard cutoff.</p>`,
    },
    castShadow: {
        title: 'Cast Shadow',
        body: `
<p>Whether this light generates a shadow map. When enabled, the engine
renders depth-only passes from the light's point of view into the
<strong>shadow atlas</strong>.</p>
<p>Shadow-casting lights consume atlas space and require additional
draw calls per frame (one per cascade/face).</p>
<p class="hint-note">The shadow type is chosen automatically based on the light type
(Directional &rarr; CSM, Spot &rarr; Standard, Point &rarr; Cube)
unless explicitly overridden.</p>`,
    },
    shadowBias: {
        title: 'Shadow Bias',
        body: `
<p>Offsets shadow depth comparisons to prevent <strong>shadow acne</strong>
(self-shadowing artifacts caused by depth buffer precision limits).</p>
<p>The bias is interpreted differently per shadow type:</p>
<div class="hint-values">
<div><strong>CSM (directional):</strong> dimensionless <em>texel multiplier</em>.
<code>ndcBias = bias &times; texelWorldSize / depthRange</code>.
Recommended: <code>1.0–2.0</code></div>
<div><strong>Spot / Cube:</strong> <em>world-space offset</em>.
The GPU applies depth-correct scaling:
<code>depthCorrect = rawBias &times; d&sup2; / (near &times; far)</code>
with a floor of <code>rawBias &times; 0.1</code></div>
</div>
<p>Additionally, slope-based scaling is applied:
<code>slopeBias = bias / clamp(NdotL, 0.25, 1.0)</code></p>
<p class="hint-note">Too little bias = shadow acne. Too much = peter-panning
(shadows detach from objects). The depth-correct formula for
perspective lights prevents the extreme peter-panning that a
constant NDC bias would cause at distance.</p>`,
    },
    pcfRadius: {
        title: 'PCF Radius',
        body: `
<p>Controls the <strong>Percentage Closer Filtering</strong> kernel size,
which softens shadow edges by averaging multiple depth comparisons.</p>
<div class="hint-values">
<div><code>0</code> — 1 tap, no filtering (hard shadows)</div>
<div><code>1</code> — 3&times;3 kernel, 9 taps (default, subtle softness)</div>
<div><code>2</code> — 5&times;5 kernel, 25 taps (soft shadows)</div>
<div><code>3</code> — 7&times;7 kernel, 49 taps (very soft, most expensive)</div>
</div>
<p>The shader loops <code>[-pcfR, +pcfR]</code> in both X and Y,
sampling the shadow atlas with a comparison sampler at each offset.
Taps are clamped to the atlas tile bounds to prevent bleeding
between lights.</p>
<p class="hint-note">Higher radius = softer shadows but more texture samples per pixel.
Going from 1 to 3 increases taps from 9 to 49 (5.4&times; more work).</p>`,
    },
    shadowType: {
        title: 'Shadow Type',
        body: `
<p>The shadow mapping algorithm used for this light.</p>
<div class="hint-values">
<div><strong>None</strong> (0) — no shadows</div>
<div><strong>Standard</strong> (1) — single perspective shadow map.
Default for <em>spot lights</em>. 1 atlas tile.</div>
<div><strong>Cascaded</strong> (2) — Cascaded Shadow Maps (CSM) with
log-linear split (&lambda;=0.85). Default for <em>directional lights</em>.
1&ndash;4 atlas tiles, texel-snapped to prevent shadow swimming.</div>
<div><strong>Cube</strong> (3) — 6-face omnidirectional shadow map.
Default for <em>point lights</em>. 6 atlas tiles. Face-seam blending
with smoothstep over 12% edge zone.</div>
</div>`,
    },
    shadowResolution: {
        title: 'Shadow Map Resolution',
        body: `
<p>Size in pixels of each shadow map tile in the atlas. Applied per
cascade/face — a CSM with 3 cascades at 1024px uses three
1024&times;1024 atlas tiles.</p>
<div class="hint-values">
<div><code>256</code> — low quality, minimal atlas usage</div>
<div><code>512</code> — default, good balance</div>
<div><code>1024</code> — high quality, recommended for hero lights</div>
<div><code>2048</code> — very high quality, expensive atlas usage</div>
</div>
<p>The atlas uses a strip allocator — tiles are packed left-to-right,
top-to-bottom. If total tile area exceeds atlas size, some lights
won't get shadows.</p>
<p class="hint-note">Resolution also affects CSM bias: the texel world size is
<code>2R / resolution</code>, so higher resolution reduces the
bias needed to prevent acne.</p>`,
    },
    cascades: {
        title: 'Cascade Count (CSM)',
        body: `
<p>Number of depth slices for <strong>Cascaded Shadow Maps</strong> (1&ndash;4).
Each cascade covers a different depth range of the camera frustum.</p>
<p>Split positions use a <strong>log-linear blend</strong> (&lambda;=0.85):</p>
<p><code>split = 0.85 &times; log + 0.15 &times; linear</code></p>
<p>This allocates more resolution to nearby surfaces where shadow
detail matters most.</p>
<div class="hint-values">
<div><code>1</code> — single shadow map, simplest but low quality at distance</div>
<div><code>2</code> — near/far split, good for small scenes</div>
<div><code>3</code> — default, good balance of quality vs. cost</div>
<div><code>4</code> — highest quality, 4 atlas tiles per light</div>
</div>
<p>Cascade boundaries are blended (last 15% of each range) using
smoothstep to prevent visible seams between cascades.</p>
<p class="hint-note">Each cascade is texel-snapped (quantized to texel grid) to
prevent shadow swimming when the camera moves.</p>`,
    },
};
