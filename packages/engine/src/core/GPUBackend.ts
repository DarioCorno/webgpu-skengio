// /src/engine/core/GPUBackend.ts

import { Logger } from './Logger';

/**
 * Feature tiers the engine can operate at.
 * Base    = standard raster only
 * Tier1   = compute shaders, indirect draw
 * Tier2   = RT extensions (future)
 */
export enum FeatureTier {
    Base = 0,
    Tier1 = 1,
    Tier2_RT = 2,
}

/**
 * Configuration supplied when initialising the backend.
 */
export interface GPUBackendConfig {
    canvas: HTMLCanvasElement;
    powerPreference?: GPUPowerPreference;
    requiredFeatures?: GPUFeatureName[];
    preferredFormat?: GPUTextureFormat;
    maxFramesInFlight?: number; // double / triple buffering (default 2)
}

/**
 * Holds everything the rest of the engine needs from the GPU:
 * adapter, device, queue, surface/swapchain, and capability queries.
 */
export class GPUBackend {

    // --- public handles -------------------------------------------------------
    adapter!: GPUAdapter;
    device!: GPUDevice;
    queue!: GPUQueue;
    context!: GPUCanvasContext;
    preferredFormat!: GPUTextureFormat;
    featureTier: FeatureTier = FeatureTier.Base;
    maxFramesInFlight: number = 2;
    /** True when the device was created with the 'timestamp-query' feature. */
    hasTimestampQuery: boolean = false;

    // --- private state --------------------------------------------------------
    private _canvas!: HTMLCanvasElement;
    private _lost: boolean = false;
    private readonly _log = new Logger('GPUBackend');

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    /**
     * Async factory – requests adapter & device, configures the surface.
     */
    async init(config: GPUBackendConfig): Promise<void> {
        this._canvas = config.canvas;
        this.maxFramesInFlight = config.maxFramesInFlight ?? 2;

        if (!navigator.gpu) {
            throw new Error('[GPUBackend] WebGPU is not supported in this browser.');
        }

        const powerPreference = config.powerPreference ?? 'high-performance';
        const adapter = await navigator.gpu.requestAdapter({ powerPreference });
        if (!adapter) throw new Error('[GPUBackend] Failed to obtain GPUAdapter.');
        this.adapter = adapter;
        this._log.info('Adapter obtained', { powerPreference });

        const requiredFeatures: GPUFeatureName[] = [...(config.requiredFeatures ?? [])];

        // Auto-request timestamp-query for GPU timing if the adapter supports it.
        if (adapter.features.has('timestamp-query') && !requiredFeatures.includes('timestamp-query')) {
            requiredFeatures.push('timestamp-query');
        }

        this.device = await adapter.requestDevice({ requiredFeatures });
        this.hasTimestampQuery = this.device.features.has('timestamp-query');
        this.device.lost.then((info) => this._onDeviceLost(info));
        this.queue = this.device.queue;
        this._log.info('Device ready', {
            requiredFeatures,
            limits: {
                maxTextureDimension2D: this.device.limits.maxTextureDimension2D,
                maxBufferSize: this.device.limits.maxBufferSize,
                maxBindGroups: this.device.limits.maxBindGroups,
            },
        });

        const context = config.canvas.getContext('webgpu');
        if (!context) throw new Error('[GPUBackend] Failed to get WebGPU canvas context.');
        this.context = context;

        this.preferredFormat = config.preferredFormat ?? navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: this.preferredFormat,
            alphaMode: 'opaque',
        });
        this._log.info(`Canvas configured — format: ${this.preferredFormat}, alphaMode: opaque`);

        if (adapter.features.has('indirect-first-instance')) {
            this.featureTier = FeatureTier.Tier1;
        }
        this._log.info(`Feature tier: ${FeatureTier[this.featureTier]}`);
    }

    /**
     * Returns the current swap-chain texture view for this frame.
     */
    getCurrentTextureView(): GPUTextureView {
        return this.context.getCurrentTexture().createView();
    }

    /**
     * Query whether a specific feature / extension is available.
     */
    supportsFeature(feature: GPUFeatureName): boolean {
        return this.device.features.has(feature);
    }

    /**
     * True if the device supports raytracing extensions.
     */
    supportsRayTracing(): boolean {
        return this.featureTier >= FeatureTier.Tier2_RT;
    }

    /**
     * Resize the swap-chain when the canvas dimensions change.
     */
    resize(_width: number, _height: number): void {
        this.context.configure({
            device: this.device,
            format: this.preferredFormat,
            alphaMode: 'opaque',
        });
    }

    /**
     * Destroy device, release all GPU objects.
     */
    destroy(): void {
        this.device.destroy();
    }

    // -------------------------------------------------------------------------
    // Internal
    // -------------------------------------------------------------------------

    private _onDeviceLost(info: GPUDeviceLostInfo): void {
        this._lost = true;
        // TODO: emit event, attempt recovery / reinit
        this._log.error(`Device lost (reason: ${info.reason}) — ${info.message}`);
    }
}