import type { HintContent } from '../composables/useHintPanel';

// Omit anchorY — it's computed dynamically from the mouse position.
type HintDef = Omit<HintContent, 'anchorY'>;

export const materialHints: Record<string, HintDef> = {
    baseColor: {
        title: 'Base Color (RGBA)',
        body: `
<p>The <strong>albedo</strong> of the surface in linear sRGB. This is the main color that the surface
reflects under white light.</p>
<p><strong>RGB</strong> channels control the reflected color. <strong>Alpha</strong> controls transparency
when the material's alpha mode is set to <code>BLEND</code> or acts as the
threshold source for <code>MASK</code> mode.</p>
<p class="hint-note">For metals (metallic &gt; 0.5), base color also tints specular
reflections via the Fresnel F0 term.</p>`,
    },
    metallic: {
        title: 'Metallic',
        body: `
<p>Controls whether the surface is a <strong>dielectric</strong> (0.0) or a <strong>conductor/metal</strong> (1.0).</p>
<p>This is a core PBR parameter that fundamentally changes how light interacts with
the surface:</p>
<div class="hint-values">
<div><code>0.0</code> — dielectric (plastic, wood, stone). Diffuse-dominant, white specular.</div>
<div><code>1.0</code> — pure metal (gold, copper, iron). No diffuse, colored specular from base color.</div>
<div><code>0.0–0.3</code> — common range for non-metals</div>
</div>
<p class="hint-note">Encoded into the G-Buffer's metallic channel (RT2.r). The upper/lower
half of the 8-bit range also carries the <em>Env Reflections</em> flag.</p>`,
    },
    roughness: {
        title: 'Roughness',
        body: `
<p>Controls the <strong>microsurface scattering</strong> of the material. Determines how blurry
or sharp reflections appear.</p>
<div class="hint-values">
<div><code>0.0</code> — perfectly smooth mirror (sharp reflections)</div>
<div><code>1.0</code> — completely rough (fully diffused reflections)</div>
<div><code>0.3–0.6</code> — typical range for most real-world materials</div>
</div>
<p>In the shader, roughness is squared (<code>&alpha; = roughness&sup2;</code>) to produce the
GGX/Trowbridge-Reitz normal distribution. This gives more perceptual linearity
when adjusting the slider.</p>
<p class="hint-note">SSR is disabled for pixels above <code>roughnessCutoff</code> (default 0.4)
since rough reflections need many rays to resolve.</p>`,
    },
    emissive: {
        title: 'Emissive (RGB)',
        body: `
<p>Light <strong>emitted</strong> by the surface, added on top of all reflected light.
Values are in <strong>linear HDR</strong> — they can exceed 1.0 for bloom-inducing glow.</p>
<p>Emissive light does <em>not</em> illuminate other objects (it is not a light source).
It bypasses all shading calculations and is added directly to the final color:</p>
<p><code>finalColor = ambient + directLighting + envIBL + emissive</code></p>
<p class="hint-note">Emissive is stored in the G-Buffer (RT2.gba) and read during the deferred
lighting pass. Values above 1.0 will appear as over-bright and can drive
any future bloom post-process.</p>`,
    },
    normalScale: {
        title: 'Normal Scale',
        body: `
<p>Multiplier applied to the <strong>tangent-space normal map</strong> perturbation.
Controls how pronounced the surface bumps appear.</p>
<div class="hint-values">
<div><code>0.0</code> — flat surface (normal map has no effect)</div>
<div><code>1.0</code> — full strength (default)</div>
<div><code>&gt;1.0</code> — exaggerated bumps</div>
</div>
<p>The G-Buffer shader scales the XY components of the sampled normal map
by this value before constructing the TBN-space normal. The Z component is
recomputed to maintain unit length.</p>
<p class="hint-note">Only has a visible effect when a Normal texture map is assigned.</p>`,
    },
    occlusionStrength: {
        title: 'Occlusion Strength',
        body: `
<p>Controls how much the <strong>ambient occlusion map</strong> darkens indirect/ambient lighting
in crevices and corners.</p>
<div class="hint-values">
<div><code>0.0</code> — AO map has no effect</div>
<div><code>1.0</code> — full AO darkening (default)</div>
</div>
<p>Applied as: <code>ambient *= mix(1.0, aoSample, occlusionStrength)</code></p>
<p>This is a <em>baked/texture</em> AO multiplier, separate from the real-time SSAO
post-process effect.</p>
<p class="hint-note">Only has a visible effect when an AO texture map is assigned.</p>`,
    },
    alphaCutoff: {
        title: 'Alpha Cutoff',
        body: `
<p>Threshold for <strong>alpha masking</strong>. Pixels with alpha below this value are
discarded in the fragment shader.</p>
<div class="hint-values">
<div><code>0.5</code> — default threshold (standard for foliage, fences, etc.)</div>
<div><code>0.0</code> — nothing is discarded</div>
<div><code>1.0</code> — everything is discarded</div>
</div>
<p>Only active when the material's Alpha mode is set to <code>MASK</code>.
The shader uses <code>if (alpha &lt; alphaCutoff) { discard; }</code></p>
<p class="hint-note">Alpha-masked materials use a separate G-Buffer pipeline variant
(<code>ALPHA_MASK=1</code>) that enables the discard instruction.</p>`,
    },
    doubleSided: {
        title: 'Double Sided',
        body: `
<p>When <strong>enabled</strong>, both front and back faces of the mesh are rendered.
When disabled, back-facing triangles are culled by the GPU rasterizer.</p>
<p>For double-sided materials, the G-Buffer shader flips the surface normal
for back-facing fragments (<code>if (!frontFacing) N = -N</code>) so lighting
is correct from both sides.</p>
<p class="hint-note">The normal flip is guarded behind <code>#ifdef DOUBLE_SIDED</code>.
Unconditional flipping on single-sided meshes causes bright streaks at
triangle silhouette edges due to <code>frontFacing</code> precision flickering
at grazing angles.</p>`,
    },
    textures: {
        title: 'Texture Maps',
        body: `
<p>Texture maps provide per-pixel detail for material properties. Each slot
overrides the corresponding uniform value:</p>
<div class="hint-values">
<div><strong>Base Color</strong> — albedo map (sRGB), multiplied by the base color factor</div>
<div><strong>Normal</strong> — tangent-space normal map (linear), uses precomputed Mikktspace tangents</div>
<div><strong>Metal/Rough</strong> — packed: blue=metallic, green=roughness (glTF convention)</div>
<div><strong>AO</strong> — ambient occlusion map (R channel), darkens indirect light</div>
<div><strong>Emissive</strong> — emission map (sRGB), multiplied by emissive factor</div>
</div>
<p class="hint-note">All maps are sampled in the G-Buffer pass and stored in render targets.
The Metal/Rough map follows the glTF packing convention where the
blue channel holds metallic and green holds roughness.</p>`,
    },

    // ── Transparency parameters (BLEND mode) ─────────────────────────

    opacity: {
        title: 'Opacity',
        body: `
<p>Overall surface transparency. Controls how much of the background
is visible through the material.</p>
<div class="hint-values">
<div><code>0.0</code> — fully transparent (invisible, fragments with alpha &lt; 0.001
are discarded)</div>
<div><code>0.5</code> — semi-transparent (50% background visible)</div>
<div><code>1.0</code> — fully opaque surface color (but still rendered in
the forward pass with refraction)</div>
</div>
<p>The final pixel blends the refracted background with the surface's
own PBR lighting, weighted by opacity and Fresnel (IOR).</p>
<p class="hint-note">Only active when Alpha Mode is set to <code>BLEND</code>.
Opaque and Mask materials ignore this parameter.</p>`,
    },
    ior: {
        title: 'Index of Refraction (IOR)',
        body: `
<p>Controls two visual effects for transparent materials:</p>
<p><strong>1. Screen-space refraction</strong> — the background behind the
surface is distorted. Higher IOR = stronger distortion.</p>
<div class="hint-values">
<div><code>1.0</code> — air (no distortion, no Fresnel)</div>
<div><code>1.33</code> — water</div>
<div><code>1.5</code> — glass (default)</div>
<div><code>2.42</code> — diamond (strong distortion)</div>
</div>
<p><strong>2. Fresnel effect</strong> — edges of the surface become more
reflective/opaque at grazing angles. Derived from IOR via
<code>F0 = ((ior &minus; 1) / (ior + 1))&sup2;</code></p>
<p><strong>3. Specular F0</strong> — the IOR also sets the dielectric specular
reflectance at normal incidence, replacing the fixed 0.04 default.</p>
<p class="hint-note">Only active when Alpha Mode is <code>BLEND</code>.
The refraction samples a snapshot of the post-processed scene taken
before the transparent pass.</p>`,
    },
    castShadow: {
        title: 'Cast Shadow',
        body: `
<p>Whether this transparent material writes to the shadow map.
When <strong>enabled</strong>, the shadow pass renders this object using a
special fragment shader that discards fragments based on opacity
and shadow opacity.</p>
<p>When <strong>disabled</strong>, the object is completely invisible to
shadow-casting lights — it casts no shadow at all.</p>
<p class="hint-note">Opaque materials always cast shadows. This toggle only
applies to <code>BLEND</code> mode materials where you may want
fully transparent objects (e.g. thin glass) to not block light.</p>`,
    },
    shadowOpacity: {
        title: 'Shadow Opacity',
        body: `
<p>Controls the <strong>darkness</strong> of the shadow cast by this
transparent surface. Uses stochastic dithering in the shadow pass:
fragments are randomly discarded based on this value, and PCF
filtering averages the result into a smooth partial shadow.</p>
<div class="hint-values">
<div><code>0.0</code> — no shadow (all fragments discarded)</div>
<div><code>0.3</code> — faint shadow (~30% of fragments pass)</div>
<div><code>0.7</code> — fairly dark shadow</div>
<div><code>1.0</code> — full solid shadow (default, no fragments discarded)</div>
</div>
<p>The shadow darkness is proportional to this value. Higher PCF
radius (set on the light) produces smoother results by averaging
more samples.</p>
<p class="hint-note">The stochastic pattern uses a screen-space hash for spatial
noise. At PCF radius 0 (1 tap) the shadow will look noisy —
use PCF radius 1+ for smooth partial shadows.</p>`,
    },
};
