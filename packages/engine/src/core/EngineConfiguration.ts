// /src/engine/core/EngineConfiguration.ts
//
// Centralised performance/quality configuration for the engine.
// Every field has a sensible default.  Scene JSON files can override
// any subset via a top-level "engineConfig" block.

// -------------------------------------------------------------------------
// Interface
// -------------------------------------------------------------------------

/**
 * Engine-wide rendering configuration.
 *
 * Holds every knob that trades quality for performance.  Defaults target
 * a mid-range discrete GPU at 1080p.  Scene JSON files can supply a
 * partial override that is merged on top of the defaults at load time.
 *
 * Fields marked *init-time* take effect when the subsystem is (re-)created
 * (engine boot or scene reload via `clearScene()`).  Fields marked
 * *runtime* can be changed between frames.
 */
export interface EngineConfiguration {

    // -----------------------------------------------------------------
    // -----------------------------------------------------------------
    // Shadows
    // -----------------------------------------------------------------

    /**
     * Shadow atlas texture size in pixels (width = height).
     * All shadow maps are packed into a single depth texture of this size.
     * Larger = sharper shadows but more VRAM (depth32float: size² × 4 B).
     *
     * *init-time* — rebuilds the atlas on scene reload.
     * @default 4096
     */
    shadowAtlasSize: number;

    /**
     * Default shadow map resolution per cascade / cube face in pixels.
     * Individual lights can still override via `shadowMapResolution` in
     * the scene JSON node.
     *
     * *init-time*
     * @default 512
     */
    defaultShadowMapResolution: number;

    /**
     * Default number of CSM cascades for directional lights (1–4).
     * More cascades = sharper near-shadows but more draw passes.
     * Individual lights can override via `numCascades`.
     *
     * *init-time*
     * @default 3
     */
    defaultCsmCascades: number;

    /**
     * Default shadow depth bias.
     * For CSM (directional): dimensionless texel multiplier (recommended 1.0–2.0).
     * For spot/cube: world-space offset (recommended 0.003–0.01).
     * Individual lights can override via `shadowBias`.
     *
     * *init-time*
     * @default 1.5
     */
    defaultShadowBias: number;

    // -----------------------------------------------------------------
    // Lighting
    // -----------------------------------------------------------------

    /**
     * Maximum number of lights the engine can process per frame.
     * Determines GPU storage buffer size (80 B × maxLights).
     *
     * *init-time*
     * @default 256
     */
    maxLights: number;

    /**
     * Maximum lights assigned to a single screen-space cluster.
     * Lower = less per-pixel work; lights beyond this cap are dropped.
     *
     * *init-time*
     * @default 32
     */
    maxLightsPerCluster: number;

    // -----------------------------------------------------------------
    // Culling
    // -----------------------------------------------------------------

    /**
     * Enable frustum culling of mesh nodes via bounding spheres.
     * Disable for debugging or tiny scenes where the overhead isn't worth it.
     *
     * *runtime*
     * @default true
     */
    frustumCulling: boolean;

    /**
     * World-space tolerance added to every bounding sphere during frustum culling.
     * Objects within this distance outside the frustum are kept visible.
     * Prevents animated meshes from popping and keeps shadow casters alive
     * slightly beyond the screen edge.
     *
     * *runtime*
     * @default 2.0
     */
    frustumCullTolerance: number;

    // -----------------------------------------------------------------
    // Textures
    // -----------------------------------------------------------------

    /**
     * Maximum anisotropic filtering level for material samplers.
     * 1 = off (fastest), 4 = medium, 16 = maximum quality.
     * Higher values improve texture clarity at oblique angles but cost
     * more texture cache bandwidth.
     *
     * *init-time* — recreates the shared material sampler on scene reload.
     * @default 4
     */
    maxAnisotropy: number;

    // -----------------------------------------------------------------
    // Draw batching
    // -----------------------------------------------------------------

    /**
     * Maximum dynamic (non-instanced) draw calls per frame.
     * Determines the model-matrix uniform buffer size (draws × 256 B).
     *
     * *init-time*
     * @default 4096
     */
    maxDrawsPerFrame: number;

    /**
     * Maximum static instances (instanced draw path) per frame.
     * Determines the instance storage buffer size (instances × 64 B).
     *
     * *init-time*
     * @default 4096
     */
    maxStaticInstances: number;

    // -----------------------------------------------------------------
    // Editor / debug UI
    // -----------------------------------------------------------------

    /**
     * Show the right-hand editor/inspector panel (DEV builds only).
     *
     * *runtime*
     * @default false
     */
    showEditorPanel: boolean;

    /**
     * Show the FPS / rendering statistics overlay (DEV builds only).
     *
     * *runtime*
     * @default true
     */
    showStats: boolean;

    // -----------------------------------------------------------------
    // Post-processing
    // -----------------------------------------------------------------

    /**
     * Tonemap operator applied in the blit pass.
     * Options: 'ACES', 'AgX', 'Reinhard', 'PBR_Neutral'.
     *
     * *runtime*
     * @default 'ACES'
     */
    tonemapOperator: string;

    // -----------------------------------------------------------------
    // Resolution scaling
    // -----------------------------------------------------------------

    /**
     * When true, the engine renders at half the native resolution and
     * stretches the canvas back to full size via CSS.  This is a simple
     * performance hack that trades visual sharpness for ~4× fewer pixels.
     *
     * *init-time* — takes effect on scene load / resize.
     * @default false
     */
    halfResolution: boolean;
}

// -------------------------------------------------------------------------
// Defaults
// -------------------------------------------------------------------------

/** Returns a fresh copy of the default engine configuration. */
export function defaultEngineConfig(): EngineConfiguration {
    return {
        // Shadows
        shadowAtlasSize:            4096,
        defaultShadowMapResolution: 512,
        defaultCsmCascades:         3,
        defaultShadowBias:          1.5,

        // Lighting
        maxLights:                  256,
        maxLightsPerCluster:        32,

        // Culling
        frustumCulling:             true,
        frustumCullTolerance:       2.0,

        // Textures
        maxAnisotropy:              4,

        // Draw batching
        maxDrawsPerFrame:           4096,
        maxStaticInstances:         4096,

        // Editor / debug UI
        showEditorPanel:            false,
        showStats:                  true,

        // Post-processing
        tonemapOperator:            'ACES',

        // Resolution scaling
        halfResolution:             false,
    };
}

// -------------------------------------------------------------------------
// Merge helper
// -------------------------------------------------------------------------

/**
 * Merge a partial override on top of the current configuration.
 * Only fields present in `overrides` replace the corresponding defaults.
 */
export function mergeEngineConfig(
    base: EngineConfiguration,
    overrides: Partial<EngineConfiguration>,
): EngineConfiguration {
    return { ...base, ...overrides };
}
