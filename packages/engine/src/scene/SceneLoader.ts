// /src/engine/scene/SceneLoader.ts
//
// Loads a scene from a JSON file and populates the engine with it.
//
// Usage (in main.ts):
//   const loader = new SceneLoader();
//   await loader.load(engine, '/scenes/demo.json', canvas);
//
// ─────────────────────────────────────────────────────────────────────────────

import type { Engine } from '../Engine';
import { NodeType } from './SceneGraph';
import { LightType, ShadowType } from '../lights/LightSystem';
import { ProjectionType } from '../camera/Camera';
import { AlphaMode } from '../materials/MaterialSystem';
import { GeometryUtils } from '../geometry/GeometryUtils';
import { FreeLookController, OrbitController, EditorController } from '../input/CameraController';
import type { ResourceHandle } from '../core/ResourceManager';
import type { MaterialHandle } from '../materials/MaterialSystem';
import type { MeshHandle } from '../geometry/MeshSystem';
import { GLTFLoader } from './GLTFLoader';
import type { EngineConfiguration } from '../core/EngineConfiguration';
import { BackgroundType } from '../environment/BackgroundSystem';
import type { BackgroundConfig } from '../environment/BackgroundSystem';

// ─────────────────────────────────────────────────────────────────────────────
// JSON schema types (what the .json file may contain)
// ─────────────────────────────────────────────────────────────────────────────

/** Top-level scene document. */
interface SceneJSON {
    /**
     * Engine performance / quality configuration overrides.
     * Any field from EngineConfiguration can be set here to override the
     * engine defaults for this scene.  See EngineConfiguration for the full
     * list of available knobs and their defaults.
     */
    engineConfig?: Partial<EngineConfiguration>;
    /** Camera configuration (one per scene). */
    camera?: CameraJSON;
    /**
     * Named texture assets.  Keys are arbitrary IDs referenced by materials.
     * Values are URL paths served from /public (e.g. "/textures/foo.jpg").
     */
    textures?: Record<string, string>;
    /**
     * Named PBR materials.  Keys are arbitrary IDs referenced by node "material" fields.
     */
    materials?: Record<string, MaterialJSON>;
    /**
     * Named mesh prototypes.  Keys are arbitrary IDs that nodes and instanceGroups
     * can reference via `meshRef`.  Each prototype is uploaded to the GPU once
     * and its MeshHandle is shared across all referencing nodes, enabling
     * automatic GPU instancing for objects that share the same mesh + material.
     */
    meshPrototypes?: Record<string, MeshJSON>;
    /**
     * Environment configuration (background, future: IBL, fog, …).
     */
    envConfig?: EnvConfigJSON;
    /** Scene-graph nodes (meshes, lights, and glTF models, in any order). */
    nodes?: NodeJSON[];
    /**
     * Compact instanced object lists.  Each group specifies one mesh prototype
     * + one material and a flat list of per-instance transforms.  All instances
     * share the same MeshHandle and MaterialHandle and are automatically batched
     * into a single GPU draw call by the FrameOrchestrator.
     */
    instanceGroups?: InstanceGroupJSON[];
}

/** SSAO configuration block. */
interface SSAOConfigJSON {
    /** Enable/disable SSAO (default: true). */
    enabled?:        boolean;
    /** World-space hemisphere radius (default: 0.5). */
    radius?:         number;
    /** Depth bias to avoid self-occlusion (default: 0.02). */
    bias?:           number;
    /** AO strength multiplier (default: 1.5). */
    intensity?:      number;
    /** Samples per pixel: 8, 12, or 16 (default: 16). */
    sampleCount?:    number;
    /** Depth-edge sensitivity for bilateral blur (default: 10). */
    blurSharpness?:  number;
}

/** SSR configuration block. */
interface SSRConfigJSON {
    /** Enable/disable SSR (default: true). */
    enabled?:         boolean;
    /** Maximum ray march steps (default: 128). */
    maxRaySteps?:     number;
    /** View-space thickness in meters (default: 0.3). */
    thickness?:       number;
    /** Ray march stride (default: 2.0). */
    stride?:          number;
    /** Edge fade end (default: 0.85). */
    fadeEnd?:         number;
    /** Roughness cutoff (default: 0.7). */
    roughnessCutoff?: number;
    /** Jitter scale (default: 1.0). */
    jitterScale?:     number;
    /** World-space max ray distance (default: 50.0). */
    maxDistance?:      number;
    /** Adaptive stride depth factor (default: 0.01). */
    strideZCutoff?:   number;
    /** Cubemap fallback strength 0..1 (default: 0.5). */
    envFallbackStr?:  number;
}

/** Environment configuration block. */
interface EnvConfigJSON {
    background?: BackgroundJSON;
    /** Ambient light color (linear RGB, default [0.03, 0.03, 0.03]). */
    ambient?: [number, number, number];
    /** Screen-space ambient occlusion settings. */
    ssao?: SSAOConfigJSON;
    /** Screen-space reflections settings. */
    ssr?: SSRConfigJSON;
}

/**
 * Scene background specification.
 *
 * type: "color"    → solid color (color field)
 * type: "gradient" → vertical gradient (topColor + bottomColor)
 * type: "texture"  → 2D image (textureKey references scene.textures)
 * type: "cubemap"  → cubemap that rotates with the camera
 *                    basePath + ext → loads 6 faces: posx, negx, posy, negy, posz, negz
 */
interface BackgroundJSON {
    type: 'color' | 'gradient' | 'texture' | 'cubemap';
    /** Solid color [r, g, b] (linear). Default: [0, 0, 0]. */
    color?: [number, number, number];
    /** Gradient top color [r, g, b]. */
    topColor?: [number, number, number];
    /** Gradient bottom color [r, g, b]. */
    bottomColor?: [number, number, number];
    /** Key in scene.textures for 'texture' type. */
    textureKey?: string;
    /**
     * Base URL path for cubemap faces (without face suffix).
     * E.g. "/textures/sky/sky_" → loads sky_posx.png, sky_negx.png, etc.
     */
    basePath?: string;
    /** File extension for cubemap faces. Default: ".png". */
    ext?: string;
}

/** One instance inside an InstanceGroupJSON. */
interface InstanceTransformJSON {
    /** World-space position [x, y, z]. Default: [0, 0, 0]. */
    position?: [number, number, number];
    /** Orientation quaternion [x, y, z, w]. Default: identity. */
    rotation?: [number, number, number, number];
    /** Per-axis scale. Default: [1, 1, 1]. */
    scale?: [number, number, number];
}

/**
 * A batch of instances that all share the same mesh prototype and material.
 * Results in a single GPU instanced draw call per frame (after frustum culling).
 */
interface InstanceGroupJSON {
    /** Key in scene.meshPrototypes. */
    mesh: string;
    /** Key(s) in scene.materials.  Omit for no material. */
    material?: string | string[];
    /** Per-instance transforms. All instances are static (eligible for batching). */
    instances: InstanceTransformJSON[];
}

interface CameraJSON {
    /** World-space position [x, y, z]. Default: [0, 0, 5]. */
    position?: [number, number, number];
    /** Orientation quaternion [x, y, z, w]. Default: identity [0, 0, 0, 1]. */
    rotation?: [number, number, number, number];
    /** "perspective" | "orthographic". Default: "perspective". */
    projection?: 'perspective' | 'orthographic';
    /** Vertical field of view in degrees. Default: 60. */
    fovY?: number;
    /** Near clip plane. Default: 0.1. */
    near?: number;
    /** Far clip plane. Default: 200. */
    far?: number;
    /** EV100 exposure. Default: 0. */
    exposure?: number;
    /** Enable TAA jitter. Default: false. */
    taaEnabled?: boolean;
    /** Camera controllers to register. */
    controllers?: ControllerJSON[];
}

type ControllerType = 'freelook' | 'orbit' | 'editor';

interface ControllerJSON {
    type: ControllerType;
    /** Translation speed in world units/sec. Default: 5 (freelook), 3 (orbit). */
    moveSpeed?: number;
    /** Mouse sensitivity in radians/pixel (FreeLook). Default: 0.002. */
    lookSensitivity?: number;
    /** Orbit drag sensitivity in radians/pixel. Default: 0.005. */
    orbitSensitivity?: number;
    /** Pan drag sensitivity in world units/pixel (editor only). Default: 0.002. */
    panSensitivity?: number;
    /** Scroll zoom sensitivity (orbit/editor). Default: 0.001. */
    zoomSensitivity?: number;
    /** Sprint multiplier while Shift is held. Default: 3. */
    sprintMultiplier?: number;
    /** Initial orbit distance from target (orbit/editor). Default: 5. */
    distance?: number;
    /** Min orbit distance (orbit/editor). Default: 0.2. */
    minDistance?: number;
    /** Max orbit distance (orbit/editor). Default: 100. */
    maxDistance?: number;
    /** Target point the camera orbits around (editor only). Default: [0,0,0]. */
    target?: [number, number, number];
    /** Invert horizontal look/orbit axis. Default: false. */
    invertX?: boolean;
    /** Invert vertical look/orbit axis. Default: false. */
    invertY?: boolean;
    /** If true, this controller is activated immediately. */
    active?: boolean;
}

interface MaterialJSON {
    /** "opaque" | "mask" | "blend". Default: "opaque". */
    alphaMode?: 'opaque' | 'mask' | 'blend';
    /** Render both sides. Default: false. */
    doubleSided?: boolean;
    /** RGBA base colour / tint. Default: [1, 1, 1, 1]. */
    baseColorFactor?: [number, number, number, number];
    /** [0, 1]. Default: 0. */
    metallicFactor?: number;
    /** [0, 1]. Default: 0.5. */
    roughnessFactor?: number;
    /** Normal-map scale. Default: 1. */
    normalScale?: number;
    /** AO map strength. Default: 1. */
    occlusionStrength?: number;
    /** RGB additive emissive. Default: [0, 0, 0]. */
    emissiveFactor?: [number, number, number];
    /** Alpha cutoff for "mask" mode. Default: 0.5. */
    alphaCutoff?: number;
    /** Overall surface opacity [0,1]. 1.0 = fully opaque. Used when alphaMode == 'blend'. */
    opacity?: number;
    /** Index of refraction for Fresnel. Default: 1.5 (glass). Used when alphaMode == 'blend'. */
    ior?: number;
    /** Whether this material casts shadows. Default: true. */
    castShadow?: boolean;
    /** Shadow opacity [0,1]. 0 = no shadow, 1 = full shadow. Default: 1. */
    shadowOpacity?: number;

    // Texture references — must match a key in scene.textures.
    baseColorMap?: string;
    normalMap?: string;
    /**
     * Combined metallic-roughness map (glTF ORM convention):
     *   G channel → roughness × roughnessFactor
     *   B channel → metallic  × metallicFactor
     */
    metallicRoughnessMap?: string;
    occlusionMap?: string;
    emissiveMap?: string;
}

interface NodeJSON {
    /** Display name. */
    name?: string;
    /** World-space position [x, y, z]. Default: [0, 0, 0]. */
    position?: [number, number, number];
    /** Orientation quaternion [x, y, z, w]. Default: identity [0, 0, 0, 1]. */
    rotation?: [number, number, number, number];
    /** Uniform or per-axis scale. Default: [1, 1, 1]. */
    scale?: [number, number, number];
    /**
     * When false this node is treated as dynamic and rendered via the per-draw
     * uniform path (one draw call per object, no batching).
     *
     * Default: true (static, eligible for GPU instancing with other objects that
     * share the same mesh + material).
     *
     * Set to false for any object that moves, scales, or changes material each
     * frame.  TODO (Animations): animated nodes must be marked static: false.
     */
    static?: boolean;
    /** If present, this node renders a procedural mesh (defined inline). */
    mesh?: MeshJSON;
    /**
     * Reference to a named mesh prototype in scene.meshPrototypes.
     * Takes precedence over inline `mesh` when both are present.
     * Using meshRef lets multiple nodes share the same GPU buffers and be
     * automatically batched into instanced draw calls.
     */
    meshRef?: string;
    /**
     * Material key(s) to assign to the mesh.
     * Single string or array (one entry per mesh sub-surface).
     */
    material?: string | string[];
    /** If present, this node holds a light (no mesh rendered). */
    light?: LightJSON;
    /**
     * Animation configuration for glTF nodes with skeletal animations.
     * Only applies when the node loads a glTF model that contains animations.
     */
    animation?: AnimationJSON;
}

interface AnimationJSON {
    /** Whether animation is enabled. Default: true. Set to false to skip all animation processing. */
    enabled?: boolean;
    /**
     * Which animation clip to play.
     * Can be a clip name (string) or a zero-based index into the glTF's animation list.
     * Default: 0 (first animation).
     */
    clip?: string | number;
    /** Playback speed multiplier. Default: 1.0. */
    speed?: number;
    /** Whether to loop the animation. Default: true. */
    loop?: boolean;
    /** Whether to start playing immediately. Default: true. */
    autoplay?: boolean;
}

type MeshType = 'box' | 'uvsphere' | 'icosphere' | 'plane' | 'gltf';

interface MeshJSON {
    type: MeshType;
    // box
    width?: number;
    height?: number;
    depth?: number;
    // sphere (uvsphere / icosphere)
    radius?: number;
    widthSegments?: number;   // uvsphere
    heightSegments?: number;  // uvsphere
    subdivisions?: number;    // icosphere
    // plane
    segmentsX?: number;
    segmentsZ?: number;
    // gltf
    /** URL path to a .glb/.gltf file (e.g. "/models/helmet.glb"). */
    url?: string;
}

type LightTypeStr = 'directional' | 'point' | 'spot';

interface LightJSON {
    type: LightTypeStr;
    /** Linear RGB. Default: [1, 1, 1]. */
    color?: [number, number, number];
    /** Luminous power / intensity. Default: 1. */
    intensity?: number;
    /** Maximum influence radius for point/spot. Default: 10. */
    range?: number;
    /** Spot inner (full-intensity) cone half-angle in degrees. Default: 15. */
    innerConeAngle?: number;
    /** Spot outer (zero-intensity) cone half-angle in degrees. Default: 30. */
    outerConeAngle?: number;
    /** Whether this light casts a shadow. Default: false. */
    castShadow?: boolean;
    /** Shadow depth bias. Default: 0.005. */
    shadowBias?: number;
    /**
     * Shadow algorithm. 'standard' = single depth map (spot), 'cascaded' = CSM (directional),
     * 'cube' = omnidirectional 6-face cube map (point). Defaults: cascaded/standard/cube per type.
     */
    shadowType?: 'standard' | 'cascaded' | 'cube';
    /** Number of CSM cascade slices (1–4, directional lights only). Default: 3. */
    numCascades?: number;
    /** Shadow map resolution in pixels per cascade (power of 2). Default: 512. */
    shadowMapResolution?: number;
    /** PCF filter radius: 1 = 3×3 (default), 2 = 5×5, 3 = 7×7. Higher = softer. */
    pcfRadius?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// SceneLoader
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Loads a scene from a JSON file URL and populates the given engine.
 *
 * Shows a thin progress bar overlay while loading and removes it on completion.
 *
 * Call order:
 *   1. Fetch + parse JSON
 *   2. Upload textures (one at a time for accurate progress)
 *   3. Create materials
 *   4. Create scene nodes (meshes + lights)
 *   5. Set up camera + controllers
 */
export class SceneLoader {

    /** glTF loaders created during load — exposed for editor animation inspection. */
    readonly gltfLoaders: GLTFLoader[] = [];

    /** The raw JSON string of the last loaded scene (for the editor). */
    lastJSON = '';

    private _canvas!: HTMLCanvasElement;
    private _engine!: Engine;

    /**
     * @param engine  Fully initialised engine (after `await engine.init()`).
     * @param url     URL of the scene JSON (e.g. '/scenes/demo.json').
     * @param canvas  The render canvas (needed for aspect-ratio + resize handling).
     */
    async load(engine: Engine, url: string, canvas: HTMLCanvasElement): Promise<void> {
        this._engine = engine;
        this._canvas = canvas;

        const overlay = _createOverlay();
        const setProgress = _progressFn(overlay);

        // 1. Fetch the scene file.
        let resp: Response;
        try {
            resp = await fetch(url);
        } catch (err) {
            _showOverlayError(overlay, `Failed to fetch scene file:\n${url}\n\n${err instanceof Error ? err.message : String(err)}`);
            return;
        }

        if (!resp.ok) {
            _showOverlayError(overlay, `Scene file not found:\n${url}\n\nHTTP ${resp.status} ${resp.statusText}`);
            return;
        }

        // 2. Parse JSON — guard against HTML error pages or malformed files.
        const text = await resp.text();
        let scene: SceneJSON;
        try {
            scene = JSON.parse(text) as SceneJSON;
        } catch {
            const preview = text.slice(0, 120).replace(/</g, '&lt;');
            const isHtml = /^\s*<!doctype|^\s*<html/i.test(text);
            const hint = isHtml
                ? 'The server returned an HTML page instead of JSON.\nCheck that the file exists in /public/scenes/.'
                : 'The file is not valid JSON.';
            _showOverlayError(overlay, `Failed to parse scene:\n${url}\n\n${hint}\n\nResponse starts with:\n${preview}`);
            return;
        }

        this.lastJSON = text;
        setProgress(0.05);

        await this._populate(engine, scene, canvas, setProgress);

        setProgress(1.0);
        _removeOverlay(overlay);
    }

    /**
     * Load a scene from a JSON string.  Stops the engine, clears all
     * scene-level data (nodes, meshes, materials, lights, cameras), then
     * repopulates from the parsed JSON and restarts the render loop.
     */
    async loadFromString(json: string): Promise<void> {
        const engine = this._engine;
        const canvas = this._canvas;
        if (!engine || !canvas) {
            throw new Error('SceneLoader: call load() first to bind engine and canvas');
        }

        this.lastJSON = json;
        const scene: SceneJSON = JSON.parse(json) as SceneJSON;

        engine.clearScene();
        this.gltfLoaders.length = 0;

        const overlay = _createOverlay();
        const setProgress = _progressFn(overlay);

        await this._populate(engine, scene, canvas, setProgress);

        setProgress(1.0);
        _removeOverlay(overlay);

        engine.start();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal: populate the engine from a parsed SceneJSON
    // ─────────────────────────────────────────────────────────────────────────

    private async _populate(
        engine: Engine,
        scene: SceneJSON,
        canvas: HTMLCanvasElement,
        setProgress: (p: number) => void,
    ): Promise<void> {
        // Apply engine configuration overrides from the scene JSON.
        const configOverrides: Partial<EngineConfiguration> = {
            ...scene.engineConfig,
        };
        if (Object.keys(configOverrides).length > 0) {
            engine.applyConfig(configOverrides);
        }

        // Half-resolution hack: render at 50% pixel size, CSS stretches back.
        if (engine.config.halfResolution) {
            canvas.style.imageRendering = 'auto';
            const dpr = window.devicePixelRatio || 1;
            const w = Math.floor(canvas.clientWidth  * dpr * 0.5);
            const h = Math.floor(canvas.clientHeight * dpr * 0.5);
            if (w > 0 && h > 0) {
                canvas.width  = w;
                canvas.height = h;
                engine.backend.resize(w, h);
            }
        }

        // Budget: textures 70 %, materials 5 %, mesh prototypes 5 %, nodes+instances 10 %, camera 10 %
        const texCount = scene.textures  ? Object.keys(scene.textures).length  : 0;
        const matCount = scene.materials ? Object.keys(scene.materials).length : 0;

        // 1. Textures ──────────────────────────────────────────────────────────
        const texHandles = new Map<string, ResourceHandle>();
        const texPaths   = new Map<string, string>();
        if (scene.textures) {
            let loaded = 0;
            await Promise.all(
                Object.entries(scene.textures).map(async ([key, path]) => {
                    const handle = await engine.resources.loadImageToTexture(path, { label: key });
                    texHandles.set(key, handle);
                    texPaths.set(key, path as string);
                    loaded++;
                    setProgress(0.05 + 0.70 * (loaded / texCount));
                }),
            );
        }
        setProgress(0.75);

        // 2. Materials ─────────────────────────────────────────────────────────
        const matHandles = new Map<string, MaterialHandle>();
        if (scene.materials) {
            let matsDone = 0;
            for (const [key, m] of Object.entries(scene.materials)) {
                let alphaMode = AlphaMode.Opaque;
                if (m.alphaMode === 'mask')  alphaMode = AlphaMode.Mask;
                if (m.alphaMode === 'blend') alphaMode = AlphaMode.Blend;

                const handle = engine.materials.createMaterial({
                    label:       key,
                    alphaMode,
                    doubleSided: m.doubleSided,
                    pbrParams: {
                        baseColorFactor:   m.baseColorFactor   ?? [1, 1, 1, 1],
                        metallicFactor:    m.metallicFactor    ?? 0.0,
                        roughnessFactor:   m.roughnessFactor   ?? 0.5,
                        normalScale:       m.normalScale       ?? 1.0,
                        occlusionStrength: m.occlusionStrength ?? 1.0,
                        emissiveFactor:    m.emissiveFactor    ?? [0, 0, 0],
                        alphaCutoff:       m.alphaCutoff       ?? 0.5,
                        opacity:           m.opacity           ?? 1.0,
                        ior:               m.ior               ?? 1.5,
                        shadowOpacity:     m.shadowOpacity     ?? 1.0,
                    },
                    castShadow: m.castShadow ?? true,
                    textures: {
                        baseColorMap:         _resolve(texHandles, m.baseColorMap),
                        normalMap:            _resolve(texHandles, m.normalMap),
                        metallicRoughnessMap: _resolve(texHandles, m.metallicRoughnessMap),
                        occlusionMap:         _resolve(texHandles, m.occlusionMap),
                        emissiveMap:          _resolve(texHandles, m.emissiveMap),
                    },
                });
                matHandles.set(key, handle);
                matsDone++;
                setProgress(0.75 + 0.10 * (matsDone / Math.max(matCount, 1)));
            }
        }
        setProgress(0.80);

        // 2b. Background (envConfig.background) ───────────────────────────────
        if (scene.envConfig?.background) {
            const bg = scene.envConfig.background;
            const bgConfig: BackgroundConfig = { type: _parseBackgroundType(bg.type) };

            switch (bgConfig.type) {
                case BackgroundType.Color:
                    bgConfig.color = bg.color ?? [0, 0, 0];
                    break;
                case BackgroundType.Gradient:
                    bgConfig.topColor    = bg.topColor    ?? [0.1, 0.1, 0.3];
                    bgConfig.bottomColor = bg.bottomColor ?? [0, 0, 0];
                    break;
                case BackgroundType.Texture: {
                    const texHandle = bg.textureKey ? texHandles.get(bg.textureKey) : undefined;
                    if (!texHandle) {
                        console.warn(`SceneLoader: background textureKey "${bg.textureKey}" not found in textures`);
                    }
                    bgConfig.textureHandle = texHandle;
                    bgConfig.texturePath = bg.textureKey ? texPaths.get(bg.textureKey) : undefined;
                    break;
                }
                case BackgroundType.Cubemap: {
                    if (bg.basePath) {
                        const ext = bg.ext ?? '.png';
                        const handle = await engine.resources.loadCubemapTexture(
                            bg.basePath, ext, { label: 'bg_cubemap' },
                        );
                        bgConfig.cubemapHandle = handle;
                        bgConfig.cubemapBasePath = bg.basePath;
                        bgConfig.cubemapExt = ext;
                    } else {
                        console.warn('SceneLoader: cubemap background missing "basePath"');
                    }
                    break;
                }
            }

            engine.background.setConfig(bgConfig);

            // Link the environment cubemap to the deferred lighting pass for
            // IBL specular reflections (automatic for all materials).
            // TODO: Replace with per-probe cubemaps when reflection probes are implemented.
            if (bgConfig.cubemapHandle !== undefined) {
                engine.orchestrator.setEnvironmentCubemap(bgConfig.cubemapHandle);
            }
        }

        // 2c. Ambient light ──────────────────────────────────────────────────
        if (scene.envConfig?.ambient) {
            const [r, g, b] = scene.envConfig.ambient;
            engine.orchestrator.setAmbientColor(r, g, b);
        }

        // 2d. SSAO ──────────────────────────────────────────────────────────
        if (scene.envConfig?.ssao !== undefined) {
            const ssaoCfg = scene.envConfig.ssao;
            const ssaoEffect = engine.postProcess.getEffect<import('../postprocess/SSAOEffect').SSAOEffect>('SSAO');
            if (ssaoEffect) {
                if (ssaoCfg.enabled !== undefined)       ssaoEffect.enabled       = ssaoCfg.enabled;
                if (ssaoCfg.radius !== undefined)        ssaoEffect.radius        = ssaoCfg.radius;
                if (ssaoCfg.bias !== undefined)          ssaoEffect.bias          = ssaoCfg.bias;
                if (ssaoCfg.intensity !== undefined)     ssaoEffect.intensity     = ssaoCfg.intensity;
                if (ssaoCfg.sampleCount !== undefined)   ssaoEffect.sampleCount   = ssaoCfg.sampleCount;
                if (ssaoCfg.blurSharpness !== undefined) ssaoEffect.blurSharpness = ssaoCfg.blurSharpness;
            }
        }

        // 2e. SSR ──────────────────────────────────────────────────────────
        if (scene.envConfig?.ssr !== undefined) {
            const ssrCfg = scene.envConfig.ssr;
            const ssrEffect = engine.postProcess.getEffect<import('../postprocess/SSREffect').SSREffect>('SSR');
            if (ssrEffect) {
                if (ssrCfg.enabled !== undefined)        ssrEffect.enabled        = ssrCfg.enabled;
                if (ssrCfg.maxRaySteps !== undefined)    ssrEffect.maxRaySteps    = ssrCfg.maxRaySteps;
                if (ssrCfg.thickness !== undefined)      ssrEffect.thickness      = ssrCfg.thickness;
                if (ssrCfg.stride !== undefined)         ssrEffect.stride         = ssrCfg.stride;
                if (ssrCfg.fadeEnd !== undefined)        ssrEffect.fadeEnd        = ssrCfg.fadeEnd;
                if (ssrCfg.roughnessCutoff !== undefined) ssrEffect.roughnessCutoff = ssrCfg.roughnessCutoff;
                if (ssrCfg.jitterScale !== undefined)    ssrEffect.jitterScale    = ssrCfg.jitterScale;
                if (ssrCfg.maxDistance !== undefined)     ssrEffect.maxDistance     = ssrCfg.maxDistance;
                if (ssrCfg.strideZCutoff !== undefined)   ssrEffect.strideZCutoff   = ssrCfg.strideZCutoff;
                if (ssrCfg.envFallbackStr !== undefined)  ssrEffect.envFallbackStr  = ssrCfg.envFallbackStr;
            }
        }

        // 3. Mesh prototypes ───────────────────────────────────────────────────
        //
        // Each named prototype is uploaded to GPU once.  Its MeshHandle is shared
        // by all nodes/instanceGroups that reference it via `meshRef`, enabling
        // the FrameOrchestrator to batch them into a single instanced draw call.
        const meshProtoHandles = new Map<string, MeshHandle>();
        if (scene.meshPrototypes) {
            const protoEntries = Object.entries(scene.meshPrototypes);
            let protoDone = 0;
            for (const [key, md] of protoEntries) {
                const meshDesc = _buildMesh(md);
                if (!meshDesc) {
                    console.warn(`SceneLoader: unknown mesh type "${md.type}" in prototype "${key}"`);
                } else {
                    meshProtoHandles.set(key, engine.meshes.createMesh(meshDesc));
                }
                protoDone++;
                setProgress(0.80 + 0.05 * (protoDone / Math.max(protoEntries.length, 1)));
            }
        }
        setProgress(0.85);

        // 4. Scene nodes ───────────────────────────────────────────────────────
        const nodeList    = scene.nodes ?? [];
        const igList      = scene.instanceGroups ?? [];
        const igTotal     = igList.reduce((s, g) => s + g.instances.length, 0);
        const totalWork   = nodeList.length + igTotal;
        let   nodesDone   = 0;

        for (const nodeDef of nodeList) {
            const pos  = new Float32Array(nodeDef.position ?? [0, 0, 0]);
            const rot  = new Float32Array(nodeDef.rotation ?? [0, 0, 0, 1]);
            const scl  = new Float32Array(nodeDef.scale    ?? [1, 1, 1]);
            const name = nodeDef.name ?? 'Node';

            if (nodeDef.light) {
                const ld        = nodeDef.light;
                const lightNode = engine.scene.createNode(name, NodeType.Light);
                engine.scene.setLocalTransform(lightNode, { position: pos, rotation: rot, scale: scl });

                const lightHandle = engine.lights.createLight({
                    label:          name,
                    type:           _parseLightType(ld.type),
                    color:          ld.color     ?? [1, 1, 1],
                    intensity:      ld.intensity ?? 1.0,
                    range:          ld.range     ?? 10.0,
                    innerConeAngle: ld.innerConeAngle !== undefined ? ld.innerConeAngle * (Math.PI / 180) : undefined,
                    outerConeAngle: ld.outerConeAngle !== undefined ? ld.outerConeAngle * (Math.PI / 180) : undefined,
                    castShadow:     ld.castShadow ?? false,
                    shadowBias:     ld.shadowBias,
                    shadowType:     ld.shadowType === 'standard' ? ShadowType.Standard
                                  : ld.shadowType === 'cascaded' ? ShadowType.Cascaded
                                  : ld.shadowType === 'cube'     ? ShadowType.Cube
                                  : undefined,
                    numCascades:         ld.numCascades,
                    shadowMapResolution: ld.shadowMapResolution,
                    pcfRadius:           ld.pcfRadius,
                });
                engine.scene.setLightComponent(lightNode, lightHandle);

            } else if (nodeDef.mesh?.type === 'gltf') {
                // glTF/GLB model — loaded via GLTFLoader under a root node.
                const gltfUrl = nodeDef.mesh.url;
                if (!gltfUrl) {
                    console.warn(`SceneLoader: gltf mesh on node "${name}" missing "url"`);
                    nodesDone++;
                    continue;
                }
                const rootNode = engine.scene.createNode(name, NodeType.Empty);
                engine.scene.setLocalTransform(rootNode, { position: pos, rotation: rot, scale: scl });

                const gltfLoader = new GLTFLoader();
                await gltfLoader.load(engine, gltfUrl, rootNode);
                this.gltfLoaders.push(gltfLoader);

                // Always register skins so the mesh renders correctly (even in bind pose).
                // The skinning pipeline + joint buffer are needed regardless of playback.
                for (let si = 0; si < gltfLoader.skins.length; si++) {
                    const skinData = gltfLoader.skins[si]!;
                    const meshNode = gltfLoader.skinNodeMap.get(si);
                    if (meshNode !== undefined) {
                        engine.animations.registerSkin(skinData, meshNode);
                    }
                }

                // Register clips and start playback only if animation is enabled.
                if (nodeDef.animation?.enabled !== false && nodeDef.animation) {
                    const clipHandles: { handle: number; name: string }[] = [];
                    for (const animData of gltfLoader.animations) {
                        const h = engine.animations.registerClip(animData);
                        clipHandles.push({ handle: h, name: animData.name });
                    }

                    if (clipHandles.length > 0) {
                        const animCfg = nodeDef.animation;
                        let clipRef: string | number = 0;
                        if (animCfg.clip !== undefined) {
                            clipRef = animCfg.clip;
                        }

                        let targetClip: number | string;
                        if (typeof clipRef === 'number') {
                            const entry = clipHandles[clipRef];
                            targetClip = entry ? entry.handle : clipHandles[0]!.handle;
                        } else {
                            targetClip = clipRef;
                        }

                        engine.animations.play(rootNode, {
                            clip:     targetClip,
                            speed:    animCfg.speed ?? 1.0,
                            loop:     animCfg.loop ?? true,
                            autoplay: animCfg.autoplay ?? true,
                        });
                    }
                }

            } else if (nodeDef.meshRef || nodeDef.mesh) {
                // Resolve mesh handle: prototype reference takes priority over inline.
                let meshHandle: MeshHandle | undefined;
                if (nodeDef.meshRef) {
                    meshHandle = meshProtoHandles.get(nodeDef.meshRef);
                    if (meshHandle === undefined) {
                        console.warn(`SceneLoader: meshRef "${nodeDef.meshRef}" not found on node "${name}"`);
                        nodesDone++;
                        continue;
                    }
                } else {
                    const meshDesc = _buildMesh(nodeDef.mesh!);
                    if (!meshDesc) {
                        console.warn(`SceneLoader: unknown mesh type "${nodeDef.mesh!.type}" on node "${name}"`);
                        nodesDone++;
                        continue;
                    }
                    meshHandle = engine.meshes.createMesh(meshDesc);
                }

                const meshNode = engine.scene.createNode(name, NodeType.Mesh);
                engine.scene.setLocalTransform(meshNode, { position: pos, rotation: rot, scale: scl });

                const matKeys = Array.isArray(nodeDef.material)
                    ? nodeDef.material
                    : nodeDef.material ? [nodeDef.material] : [];
                const mats = matKeys
                    .map(k => matHandles.get(k))
                    .filter((h): h is MaterialHandle => h !== undefined);

                engine.scene.setMeshComponent(meshNode, meshHandle, mats);

                // Apply the static flag (default: true → eligible for instancing).
                if (nodeDef.static === false) {
                    engine.scene.setNodeStatic(meshNode, false);
                }
            }
            nodesDone++;
            setProgress(0.85 + 0.10 * (nodesDone / Math.max(totalWork, 1)));
        }

        // 4b. Instance groups ──────────────────────────────────────────────────
        //
        // Each group maps to one (mesh, material) batch.  All nodes within a
        // group share the same MeshHandle so the FrameOrchestrator merges them
        // into a single drawIndexed(…, instanceCount, …) call.
        for (const grp of igList) {
            const meshHandle = meshProtoHandles.get(grp.mesh);
            if (meshHandle === undefined) {
                console.warn(`SceneLoader: instanceGroup mesh "${grp.mesh}" not found in meshPrototypes`);
                continue;
            }
            const matKeys = Array.isArray(grp.material)
                ? grp.material
                : grp.material ? [grp.material] : [];
            const mats = matKeys
                .map(k => matHandles.get(k))
                .filter((h): h is MaterialHandle => h !== undefined);

            // Create a parent grouping node for this instance group.
            const matLabel = matKeys.join('+') || 'default';
            const groupLabel = `${grp.mesh} [${matLabel}]`;
            const groupNode = engine.scene.createNode(groupLabel, NodeType.Empty);
            // Store mesh/material refs on the group node for the editor to read,
            // but keep type=Empty so it is NOT rendered as a drawable.
            const groupSceneNode = engine.scene.getNode(groupNode);
            if (groupSceneNode) {
                groupSceneNode.meshHandle = meshHandle;
                groupSceneNode.materialHandles = mats;
            }

            for (const inst of grp.instances) {
                const pos = new Float32Array(inst.position ?? [0, 0, 0]);
                const rot = new Float32Array(inst.rotation ?? [0, 0, 0, 1]);
                const scl = new Float32Array(inst.scale    ?? [1, 1, 1]);
                const node = engine.scene.createNode(grp.mesh, NodeType.Mesh, groupNode);
                engine.scene.setLocalTransform(node, { position: pos, rotation: rot, scale: scl });
                engine.scene.setMeshComponent(node, meshHandle, mats);
                // Mark as instance child so the editor restricts to transform-only.
                const sceneNode = engine.scene.getNode(node);
                if (sceneNode) sceneNode.isInstance = true;
                // isStatic=true by default — all instances are eligible for batching.
                nodesDone++;
                // Update progress bar every 50 instances to avoid overhead.
                if (nodesDone % 50 === 0) {
                    setProgress(0.85 + 0.10 * (nodesDone / Math.max(totalWork, 1)));
                }
            }
        }
        setProgress(0.95);

        // 5. Camera ────────────────────────────────────────────────────────────
        if (scene.camera) {
            const c       = scene.camera;
            const camNode = engine.scene.createNode('Camera', NodeType.Camera);
            engine.scene.setLocalTransform(camNode, {
                position: new Float32Array(c.position ?? [0, 0, 5]),
                rotation: new Float32Array(c.rotation ?? [0, 0, 0, 1]),
                scale:    new Float32Array([1, 1, 1]),
            });

            const aspect    = canvas.width / Math.max(canvas.height, 1);
            const camHandle = engine.cameras.createCamera({
                label:          'MainCamera',
                projectionType: c.projection === 'orthographic'
                    ? ProjectionType.Orthographic
                    : ProjectionType.Perspective,
                perspective: {
                    fovY:        (c.fovY ?? 60) * (Math.PI / 180),
                    aspectRatio: aspect,
                    near:        c.near     ?? 0.1,
                    far:         c.far      ?? 200,
                },
                nodeHandle:  camNode,
                taaEnabled:  c.taaEnabled ?? false,
                exposure:    c.exposure   ?? 0,
            });
            engine.cameras.setActiveCamera(camHandle);

            window.addEventListener('resize', () => {
                engine.cameras.updateProjection(
                    camHandle,
                    canvas.width / Math.max(canvas.height, 1),
                );
            });

            // Controllers
            for (const ctrlDef of c.controllers ?? []) {
                let ctrlName: string | null = null;
                if (ctrlDef.type === 'freelook') {
                    engine.cameraControllers.register(new FreeLookController({
                        moveSpeed:        ctrlDef.moveSpeed,
                        lookSensitivity:  ctrlDef.lookSensitivity,
                        sprintMultiplier: ctrlDef.sprintMultiplier,
                    }));
                    ctrlName = 'FreeLook';
                } else if (ctrlDef.type === 'orbit') {
                    engine.cameraControllers.register(new OrbitController({
                        orbitSensitivity: ctrlDef.orbitSensitivity,
                        zoomSensitivity:  ctrlDef.zoomSensitivity,
                        distance:         ctrlDef.distance,
                        minDistance:       ctrlDef.minDistance,
                        maxDistance:       ctrlDef.maxDistance,
                    }));
                    ctrlName = 'Orbit';
                } else if (ctrlDef.type === 'editor') {
                    engine.cameraControllers.register(new EditorController({
                        orbitSensitivity: ctrlDef.orbitSensitivity,
                        panSensitivity:   ctrlDef.panSensitivity,
                        zoomSensitivity:  ctrlDef.zoomSensitivity,
                        distance:         ctrlDef.distance,
                        minDistance:       ctrlDef.minDistance,
                        maxDistance:       ctrlDef.maxDistance,
                        target:           ctrlDef.target,
                    }));
                    ctrlName = 'Editor';
                }
                if (ctrlName) {
                    if (ctrlDef.invertX) engine.cameraControllers.setInvertX(ctrlName, true);
                    if (ctrlDef.invertY) engine.cameraControllers.setInvertY(ctrlName, true);
                    if (ctrlDef.active)  engine.cameraControllers.setActive(ctrlName);
                }
            }
            engine.cameraControllers.attachCamera(camNode, engine.scene);
        }

    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module helpers
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Loading overlay
// ─────────────────────────────────────────────────────────────────────────────

interface Overlay { root: HTMLDivElement; bar: HTMLDivElement; wrapper: HTMLDivElement; }

function _progressFn(overlay: Overlay): (p: number) => void {
    return (p: number) => {
        overlay.bar.style.width = `${(Math.min(p, 1) * 100).toFixed(1)}%`;
    };
}

function _createOverlay(): Overlay {
    const root = document.createElement('div');
    root.style.cssText = [
        'position:fixed', 'inset:0', 'z-index:9999',
        'display:flex', 'align-items:center', 'justify-content:center',
        'background:rgba(0,0,0,0.85)',
        'transition:opacity 0.2s ease',
    ].join(';');

    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:12px;';

    const track = document.createElement('div');
    track.style.cssText = [
        'width:260px', 'height:4px',
        'background:rgba(255,255,255,0.12)',
        'border-radius:2px', 'overflow:hidden',
    ].join(';');

    const bar = document.createElement('div');
    bar.style.cssText = [
        'height:100%', 'width:0%',
        'background:#fff',
        'border-radius:2px',
        'transition:width 0.1s linear',
    ].join(';');

    track.appendChild(bar);
    wrapper.appendChild(track);
    root.appendChild(wrapper);
    document.body.appendChild(root);
    return { root, bar, wrapper };
}

function _showOverlayError(overlay: Overlay, message: string): void {
    // Stop the progress bar and turn it red.
    overlay.bar.style.width = '100%';
    overlay.bar.style.background = '#e74c3c';

    const errorEl = document.createElement('div');
    errorEl.style.cssText = [
        'color:#e74c3c',
        'font-family:monospace',
        'font-size:13px',
        'max-width:400px',
        'text-align:center',
        'line-height:1.5',
        'word-break:break-word',
    ].join(';');
    errorEl.textContent = message;
    overlay.wrapper.appendChild(errorEl);
}

function _removeOverlay(overlay: Overlay): void {
    // Immediately stop blocking clicks while the fade-out runs.
    overlay.root.style.pointerEvents = 'none';
    overlay.root.style.opacity = '0';
    // Remove from DOM after transition; fallback timeout in case transitionend doesn't fire.
    const remove = () => overlay.root.remove();
    overlay.root.addEventListener('transitionend', remove, { once: true });
    setTimeout(remove, 400);
}

function _resolve(map: Map<string, ResourceHandle>, key?: string): ResourceHandle | undefined {
    if (key === undefined) return undefined;
    const h = map.get(key);
    if (h === undefined) console.warn(`SceneLoader: texture key "${key}" not found`);
    return h;
}

function _parseBackgroundType(t: string): BackgroundType {
    if (t === 'gradient') return BackgroundType.Gradient;
    if (t === 'texture')  return BackgroundType.Texture;
    if (t === 'cubemap')  return BackgroundType.Cubemap;
    return BackgroundType.Color;
}

function _parseLightType(t: LightTypeStr): LightType {
    if (t === 'directional') return LightType.Directional;
    if (t === 'spot')        return LightType.Spot;
    return LightType.Point;
}

function _buildMesh(md: MeshJSON): ReturnType<typeof GeometryUtils.createBox> | null {
    switch (md.type) {
        case 'box':
            return GeometryUtils.createBox({
                width:  md.width,
                height: md.height,
                depth:  md.depth,
            });
        case 'uvsphere':
            return GeometryUtils.createUVSphere({
                radius:         md.radius,
                widthSegments:  md.widthSegments,
                heightSegments: md.heightSegments,
            });
        case 'icosphere':
            return GeometryUtils.createIcoSphere({
                radius:       md.radius,
                subdivisions: md.subdivisions,
            });
        case 'plane':
            return GeometryUtils.createPlane({
                width:     md.width,
                depth:     md.depth,
                segmentsX: md.segmentsX,
                segmentsZ: md.segmentsZ,
            });
        default:
            return null;
    }
}
