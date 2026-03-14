import type { HintContent } from '../composables/useHintPanel';

type HintDef = Omit<HintContent, 'anchorY'>;

export const cameraHints: Record<string, HintDef> = {
    projection: {
        title: 'Projection Type',
        body: `
<p>Determines how the 3D scene is mapped to the 2D screen.</p>
<div class="hint-values">
<div><strong>Perspective</strong> — objects farther away appear smaller (standard 3D camera).
Uses <code>perspectiveZO()</code> with WebGPU's [0, 1] depth range.</div>
<div><strong>Orthographic</strong> — no perspective distortion, parallel lines stay parallel.
Used for 2D views, technical visualization, or UI overlays.</div>
</div>`,
    },
    fovY: {
        title: 'Field of View (Vertical)',
        body: `
<p>The vertical angle of the camera's viewing cone, in degrees.
Controls how much of the scene is visible.</p>
<div class="hint-values">
<div><code>40–50°</code> — telephoto feel, low distortion</div>
<div><code>60°</code> — default, natural look</div>
<div><code>90–110°</code> — wide-angle, more visible area but more distortion</div>
</div>
<p>Stored internally in radians. The horizontal FOV is derived
automatically from <code>fovY &times; aspectRatio</code>.</p>
<p>Used to build the perspective projection matrix via
<code>mat4.perspectiveZO(fovY, aspect, near, far)</code>.</p>`,
    },
    nearClip: {
        title: 'Near Clip Plane',
        body: `
<p>Minimum distance from the camera at which geometry is rendered.
Anything closer is clipped.</p>
<p><strong>Depth precision:</strong> The ratio <code>far / near</code> determines how
the depth buffer distributes its precision. A very small near value
(e.g. 0.001) with a large far value causes severe Z-fighting on
distant surfaces.</p>
<div class="hint-values">
<div><code>0.1</code> — default, good balance for most scenes</div>
<div><code>0.01</code> — needed for close-up detail, but reduces far-range precision</div>
</div>
<p class="hint-note">Shadow mapping also depends on near/far: perspective shadow maps
use depth-correct bias scaling based on <code>near &times; far</code>. Extreme
ratios can cause peter-panning or shadow acne.</p>`,
    },
    farClip: {
        title: 'Far Clip Plane',
        body: `
<p>Maximum distance from the camera at which geometry is rendered.
Anything beyond is clipped and not drawn.</p>
<p>Keep this as small as your scene allows to maximize depth buffer
precision. A <code>far/near</code> ratio above 10,000 will cause
visible Z-fighting artifacts.</p>
<div class="hint-values">
<div><code>100</code> — small indoor scenes</div>
<div><code>1000</code> — default, typical for medium environments</div>
<div><code>10000+</code> — large outdoor worlds (consider logarithmic depth)</div>
</div>
<p class="hint-note">The far plane also defines the sky sentinel: pixels at
<code>depth &ge; 1.0</code> in the G-Buffer are treated as background
and skipped by the deferred lighting pass.</p>`,
    },
    orthoLR: {
        title: 'Left / Right Bounds',
        body: `
<p>The horizontal extent of the orthographic view volume in world units.</p>
<p>Geometry outside <code>[left, right]</code> in camera-local X is clipped.
The visible width equals <code>right &minus; left</code>.</p>
<p>Built into the projection matrix via
<code>mat4.orthoZO(left, right, bottom, top, near, far)</code>.</p>
<p class="hint-note">Also used internally by CSM (cascaded shadow maps) for directional
light projections — each cascade gets its own fitted ortho bounds.</p>`,
    },
    orthoBT: {
        title: 'Bottom / Top Bounds',
        body: `
<p>The vertical extent of the orthographic view volume in world units.</p>
<p>Geometry outside <code>[bottom, top]</code> in camera-local Y is clipped.
The visible height equals <code>top &minus; bottom</code>.</p>`,
    },
    orthoNF: {
        title: 'Near / Far Planes (Ortho)',
        body: `
<p>Depth range of the orthographic view volume.</p>
<p>Unlike perspective, orthographic depth is <strong>linear</strong>, so
precision is distributed evenly across [near, far]. Z-fighting is
less of a concern, but keep the range tight for best results.</p>
<div class="hint-values">
<div>Default: <code>near = 0.1</code>, <code>far = 100</code></div>
</div>`,
    },
    exposure: {
        title: 'Exposure (EV100)',
        body: `
<p>Physically-based camera exposure using the <strong>EV100</strong> scale
(Lagarde/de Rousiers model). Controls overall image brightness
before tonemapping.</p>
<p>Converted to a linear multiplier:
<code>exposure = 1 / (2<sup>EV100</sup> &times; 1.2)</code></p>
<div class="hint-values">
<div><code>0</code> — default (multiplier &asymp; 0.83)</div>
<div><code>negative</code> — brighter image (more light gathered)</div>
<div><code>positive</code> — darker image (less light gathered)</div>
</div>
<p>Applied in the <strong>blit/tonemap shader</strong> as the final step
before ACES tonemapping:</p>
<p><code>ldr = acesTonemap(hdrColor &times; exposure)</code></p>
<p class="hint-note">Exposure is a post-process multiplier — it does not affect
lighting calculations, shadows, or G-Buffer values. Only the
final HDR-to-LDR conversion is scaled.</p>`,
    },
    taa: {
        title: 'Temporal Anti-Aliasing (TAA)',
        body: `
<p>Reduces jagged edges by <strong>jittering</strong> the camera sub-pixel
position each frame and blending results over time.</p>
<p>When enabled:</p>
<div class="hint-values">
<div>A <strong>Halton sequence</strong> (base 2, 3) generates 64 unique
sub-pixel offsets in [&minus;0.5, 0.5] pixel range</div>
<div>Jitter is applied to the projection matrix each frame</div>
<div>A velocity buffer tracks per-pixel motion for reprojection</div>
<div>History clamping (3&times;3 neighbourhood) prevents ghosting</div>
</div>
<p>When disabled: no jitter, no temporal blending. The image may
show hard aliasing on edges but has zero motion artifacts.</p>
<p class="hint-note">TAA also improves the quality of SSAO temporal accumulation
by providing stable velocity data for disocclusion detection.</p>`,
    },
    controller: {
        title: 'Camera Controller',
        body: `
<p>Selects which camera controller drives the viewport camera.</p>
<div class="hint-values">
<div><strong>FreeLook</strong> — first-person fly camera.
WASD to move, mouse to look. Shift to sprint. E/Space up, Q down.
Requires pointer lock (click viewport to capture cursor).</div>
<div><strong>Orbit</strong> — turntable around the scene origin.
LMB drag to orbit, scroll wheel to zoom. No keyboard.</div>
<div><strong>Editor</strong> — 3D-editor style (Maya/Blender).
LMB drag to pan, RMB drag to orbit, scroll to dolly.
Orbits a moveable target point.</div>
</div>
<p>The scene JSON <code>controllers</code> array determines which are
loaded with custom settings. All three built-in controllers are
always available in the dropdown regardless.</p>`,
    },
    invertAxes: {
        title: 'Invert Axes',
        body: `
<p>Inverts the horizontal or vertical look/orbit axis for the
active camera controller. Applies to both mouse and gamepad input.</p>
<div class="hint-values">
<div><strong>Invert X</strong> — reverses horizontal rotation.
Mouse drag right rotates left, gamepad right stick right rotates left.</div>
<div><strong>Invert Y</strong> — reverses vertical rotation.
Mouse drag down looks up, gamepad right stick down looks up
(classic "inverted Y" flight-sim style).</div>
</div>
<p>Per-controller setting: each controller remembers its own inversion
state independently. Can also be set in the scene JSON via
<code>invertX</code> / <code>invertY</code> on a controller definition.</p>`,
    },
    device: {
        title: 'Input Device',
        body: `
<p>Selects which input device drives the active camera controller.</p>
<div class="hint-values">
<div><strong>Mouse</strong> — keyboard + mouse input (default).
Each controller has its own mouse/keyboard mapping.</div>
<div><strong>Gamepad</strong> — browser Gamepad API (standard mapping).
Left stick and right stick control varies by controller:</div>
</div>
<p><strong>FreeLook:</strong> left stick = move forward/back &amp; strafe,
right stick = look rotation.</p>
<p><strong>Orbit:</strong> left stick Y = zoom in/out,
right stick = orbit rotation.</p>
<p><strong>Editor:</strong> left stick X = horizontal pan,
left stick Y = zoom in/out, right stick = orbit rotation.</p>
<p class="hint-note">A gamepad must be connected and have a standard mapping.
The first connected gamepad is used. A deadzone of 0.15 is
applied to all axes.</p>`,
    },
};
