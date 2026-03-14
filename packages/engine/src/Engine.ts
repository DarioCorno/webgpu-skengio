// /src/engine/Engine.ts

import { GPUBackend, type GPUBackendConfig } from './core/GPUBackend';
import { ResourceManager } from './core/ResourceManager';
import { ShaderSystem } from './shaders/ShaderSystem';
import { PipelineManager } from './pipelines/PipelineManager';
import { MeshSystem } from './geometry/MeshSystem';
import { MaterialSystem } from './materials/MaterialSystem';
import { SceneGraph } from './scene/SceneGraph';
import { CameraSystem } from './camera/Camera';
import { LightSystem } from './lights/LightSystem';
import { RenderGraph } from './rendergraph/RenderGraph';
import { FrameOrchestrator } from './commands/FrameOrchestrator';
import { PostProcessStack } from './postprocess/PostProcessStack';
import { InputSystem } from './input/InputSystem';
import { CameraControllerSystem } from './input/CameraController';
import { BackgroundSystem } from './environment/BackgroundSystem';
import { AnimationSystem } from './animation/AnimationSystem';
import { defaultEngineConfig, mergeEngineConfig, type EngineConfiguration } from './core/EngineConfiguration';

// -------------------------------------------------------------------------
// Engine configuration
// -------------------------------------------------------------------------

export interface EngineConfig {
    canvas: HTMLCanvasElement;
    powerPreference?: GPUPowerPreference;
    maxFramesInFlight?: number;
}

// -------------------------------------------------------------------------
// Update callback types
// -------------------------------------------------------------------------

/** Context passed to user update callbacks every frame. */
export interface UpdateContext {
    /** Seconds since last frame. */
    deltaTime:  number;
    /** Total elapsed time in seconds since engine.start(). */
    time:       number;
    /** Current frame index (monotonically increasing). */
    frameIndex: number;
    /** The engine instance — gives full access to all subsystems. */
    engine:     Engine;
}

/** A per-frame update function registered via engine.onUpdate(). */
export type UpdateCallback = (ctx: UpdateContext) => void;

/**
 * Top-level facade that owns every subsystem and provides
 * a single entry point for the application.
 *
 * Usage:
 *
 *   const engine = new Engine();
 *   await engine.init({ canvas: myCanvas });
 *   // … populate scene, materials, meshes …
 *   engine.start();         // begins rAF loop
 *   // …
 *   engine.stop();
 *   engine.destroy();
 */
export class Engine {

    // --- subsystems (public for direct access when needed) --------------------
    readonly backend            = new GPUBackend();
    readonly resources          = new ResourceManager();
    readonly shaders            = new ShaderSystem();
    readonly pipelines          = new PipelineManager();
    readonly meshes             = new MeshSystem();
    readonly materials          = new MaterialSystem();
    readonly scene              = new SceneGraph();
    readonly cameras            = new CameraSystem();
    readonly lights             = new LightSystem();
    readonly renderGraph        = new RenderGraph();
    readonly postProcess        = new PostProcessStack();
    readonly orchestrator       = new FrameOrchestrator();
    readonly background         = new BackgroundSystem();
    readonly animations         = new AnimationSystem();
    readonly input              = new InputSystem();
    readonly cameraControllers  = new CameraControllerSystem();

    /** Active engine configuration (defaults + scene overrides). */
    config: EngineConfiguration = defaultEngineConfig();

    private _canvas!: HTMLCanvasElement;
    private _updateCallbacks: UpdateCallback[] = [];
    private _elapsedTime = 0;

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    /**
     * Async initialisation — must be awaited before anything else.
     * Requests the GPU adapter/device, then boots every subsystem in order.
     */
    async init(config: EngineConfig): Promise<void> {
        this._canvas = config.canvas;

        // 1. GPU backend (async — adapter + device request)
        await this.backend.init({
            canvas: config.canvas,
            powerPreference: config.powerPreference ?? 'high-performance',
            maxFramesInFlight: config.maxFramesInFlight ?? 2,
        });

        // 2. Resource manager (depends on backend)
        this.resources.init(this.backend);

        // 3. Shader system (depends on backend)
        this.shaders.init(this.backend);

        // 4. Pipeline manager (depends on backend + shaders)
        this.pipelines.init(this.backend, this.shaders);

        // 5. Geometry system (depends on backend + resources)
        this.meshes.init(this.backend, this.resources);

        // 6. Material system (depends on backend + resources + shaders)
        this.materials.init(this.backend, this.resources, this.shaders, this.config);

        // 7. Scene graph (CPU only)
        this.scene.init();

        // 8. Camera system (CPU only)
        this.cameras.init();

        // 9. Light system (depends on backend + resources)
        this.lights.init(this.backend, this.resources, this.config);

        // 10. Render graph (depends on backend + resources)
        this.renderGraph.init(this.backend, this.resources);

        // 11. Post-process stack — initialized by FrameOrchestrator.init()
        //      (needs per-frame buffers, samplers, and subsystem refs created there)

        // 12. Background system (depends on backend + resources)
        this.background.init(this.backend, this.resources);

        // 12b. Animation system (depends on backend + scene graph)
        this.animations.init(this.backend, this.scene);

        // 13. Frame orchestrator (depends on everything above)
        this.orchestrator.init({
            backend:              this.backend,
            resources:            this.resources,
            shaderSystem:         this.shaders,
            pipelineManager:      this.pipelines,
            meshSystem:           this.meshes,
            renderGraph:          this.renderGraph,
            sceneGraph:           this.scene,
            cameraSystem:         this.cameras,
            lightSystem:          this.lights,
            materialSystem:       this.materials,
            postProcessStack:     this.postProcess,
            inputSystem:          this.input,
            cameraControllers:    this.cameraControllers,
            backgroundSystem:     this.background,
            animationSystem:      this.animations,
            config:               this.config,
            engine:               this,
        });

        // 14. Input system (attach event listeners to the canvas)
        this.input.init(config.canvas);

        // 14. Camera controller system (canvas reference for pointer-lock requests)
        this.cameraControllers.init(config.canvas, this.input);

        // Handle canvas resize
        this._setupResizeObserver();
    }

    // -------------------------------------------------------------------------
    // Run control
    // -------------------------------------------------------------------------

    start(): void {
        this.orchestrator.start();
    }

    stop(): void {
        this.orchestrator.stop();
    }

    // -------------------------------------------------------------------------
    // Per-frame update callbacks
    // -------------------------------------------------------------------------

    /** Register a callback that runs every frame after animations, before world matrix propagation. */
    onUpdate(fn: UpdateCallback): void {
        this._updateCallbacks.push(fn);
    }

    /** Remove a previously registered update callback. */
    offUpdate(fn: UpdateCallback): void {
        const idx = this._updateCallbacks.indexOf(fn);
        if (idx >= 0) this._updateCallbacks.splice(idx, 1);
    }

    /**
     * Called by FrameOrchestrator each frame to run user callbacks.
     * @internal
     */
    _runUpdateCallbacks(dt: number, frameIndex: number): void {
        this._elapsedTime += dt;
        if (this._updateCallbacks.length === 0) return;
        const ctx: UpdateContext = {
            deltaTime:  dt,
            time:       this._elapsedTime,
            frameIndex,
            engine:     this,
        };
        for (const fn of this._updateCallbacks) {
            fn(ctx);
        }
    }

    /**
     * Tear down all scene-level content (scene graph, meshes, materials,
     * lights, cameras, controllers) while keeping the GPU backend, resource
     * manager, shaders, pipelines, render graph, post-process, and input
     * system alive.  Call this before re-populating the scene via SceneLoader.
     */
    /**
     * Apply a partial configuration override, merging on top of defaults.
     * Call this before `clearScene()` / scene reload for full effect.
     */
    applyConfig(overrides: Partial<EngineConfiguration>): void {
        this.config = mergeEngineConfig(this.config, overrides);
    }

    clearScene(): void {
        // 1. Stop and tear down the orchestrator first (holds bind groups
        //    referencing light/shadow/mesh GPU resources that are about to die).
        this.orchestrator.destroy();

        // 2. Destroy + re-init scene-level subsystems.
        this.cameraControllers.destroy();
        this.cameraControllers.init(this._canvas, this.input);
        this.lights.destroy();
        this.lights.init(this.backend, this.resources, this.config);
        this.cameras.destroy();
        this.cameras.init();
        this.scene.destroy();
        this.scene.init();
        this.materials.destroy();
        this.materials.init(this.backend, this.resources, this.shaders, this.config);
        this.meshes.destroy();
        this.meshes.init(this.backend, this.resources);
        this.animations.destroy();
        this.animations.init(this.backend, this.scene);
        this.background.destroy();
        this.background.init(this.backend, this.resources);

        // 3. Re-init the orchestrator with fresh subsystem state.
        this.orchestrator.init({
            backend:              this.backend,
            resources:            this.resources,
            shaderSystem:         this.shaders,
            pipelineManager:      this.pipelines,
            meshSystem:           this.meshes,
            renderGraph:          this.renderGraph,
            sceneGraph:           this.scene,
            cameraSystem:         this.cameras,
            lightSystem:          this.lights,
            materialSystem:       this.materials,
            postProcessStack:     this.postProcess,
            inputSystem:          this.input,
            cameraControllers:    this.cameraControllers,
            backgroundSystem:     this.background,
            animationSystem:      this.animations,
            config:               this.config,
            engine:               this,
        });
    }

    // -------------------------------------------------------------------------
    // Resize handling
    // -------------------------------------------------------------------------

    private _setupResizeObserver(): void {
        const observer = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                const dpr = window.devicePixelRatio || 1;
                const scale = this.config.halfResolution ? 0.5 : 1;
                const w = Math.floor(width * dpr * scale);
                const h = Math.floor(height * dpr * scale);
                if (w === 0 || h === 0) continue;
                this._canvas.width = w;
                this._canvas.height = h;
                this.backend.resize(w, h);

                // Update camera aspect ratio so projection matches the new canvas size
                const cam = this.cameras.getActiveCamera();
                if (cam) {
                    this.cameras.updateProjection(cam.handle, w / h);
                }
            }
        });
        observer.observe(this._canvas);
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    destroy(): void {
        this.cameraControllers.destroy();
        this.orchestrator.destroy();
        this.animations.destroy();
        this.background.destroy();
        this.input.destroy();
        this.postProcess.destroy();
        this.renderGraph.destroy();
        this.lights.destroy();
        this.cameras.destroy();
        this.scene.destroy();
        this.materials.destroy();
        this.meshes.destroy();
        this.pipelines.destroy();
        this.shaders.destroy();
        this.resources.destroyAll();
        this.backend.destroy();
    }
}