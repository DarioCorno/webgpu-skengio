// /src/engine/rendergraph/RenderGraph.ts

import { Logger } from '../core/Logger';
import type { GPUBackend } from '../core/GPUBackend';
import type { ResourceManager, ResourceHandle, TextureDescriptor } from '../core/ResourceManager';

// -------------------------------------------------------------------------
// Virtual resource
// -------------------------------------------------------------------------

export type VirtualResourceId = number;

export enum ResourceAccess {
    Read      = 'READ',
    Write     = 'WRITE',
    ReadWrite = 'READ_WRITE',
}

export enum ResourceType {
    Texture = 'TEXTURE',
    Buffer  = 'BUFFER',
}

export interface VirtualResource {
    id: VirtualResourceId;
    name: string;
    type: ResourceType;
    textureDesc?: TextureDescriptor;
    /** Set by compile() after allocating the physical GPU resource. */
    physicalHandle?: ResourceHandle;
    /** True for resources imported from outside the graph (e.g. swap-chain). */
    external: boolean;
}

// -------------------------------------------------------------------------
// Pass types
// -------------------------------------------------------------------------

export type PassHandle = number;

export enum PassType {
    Render   = 'RENDER',
    Compute  = 'COMPUTE',
    RayTrace = 'RAYTRACE',
    Transfer = 'TRANSFER',
}

export interface PassResourceUsage {
    resourceId: VirtualResourceId;
    access: ResourceAccess;
}

/**
 * Callback invoked during execute().
 *
 * For Render passes:   passEncoder is GPURenderPassEncoder  (if attachmentIds supplied)
 * For Compute passes:  passEncoder is GPUComputePassEncoder
 * For Transfer passes: passEncoder is null (use encoder directly)
 *
 * The passEncoder is already begun when the callback is called,
 * and will be ended by the graph after the callback returns.
 * Do NOT call passEncoder.end() inside the callback.
 */
export type PassExecuteFn = (
    encoder: GPUCommandEncoder,
    passEncoder: GPURenderPassEncoder | GPUComputePassEncoder | null,
) => void;

// -------------------------------------------------------------------------
// Color / depth attachment config (per pass, per attachment)
// -------------------------------------------------------------------------

export interface ColorAttachmentConfig {
    resourceId:  VirtualResourceId;
    loadOp:      GPULoadOp;    // 'clear' | 'load'
    storeOp:     GPUStoreOp;   // 'store' | 'discard'
    clearColor?: GPUColor;     // used when loadOp='clear'; default {r:0,g:0,b:0,a:1}
    /** Optional resolve target (MSAA). */
    resolveTargetId?: VirtualResourceId;
}

export interface DepthAttachmentConfig {
    resourceId:       VirtualResourceId;
    depthLoadOp:      GPULoadOp;
    depthStoreOp:     GPUStoreOp;
    depthClearValue?: number;    // default 1.0
    /** When true the depth attachment is read-only: depthLoadOp/depthStoreOp are
     *  ignored by the GPU (content persists) and the graph does NOT derive
     *  automatic read/write edges from this attachment. */
    depthReadOnly?:   boolean;
    stencilLoadOp?:   GPULoadOp;
    stencilStoreOp?:  GPUStoreOp;
    stencilClearValue?: number;
}

// -------------------------------------------------------------------------
// Pass descriptor
// -------------------------------------------------------------------------

export interface RenderPass {
    handle:  PassHandle;
    name:    string;
    type:    PassType;
    reads:   VirtualResourceId[];
    writes:  VirtualResourceId[];

    /** Render passes: explicit color attachment bindings. */
    colorAttachments?: ColorAttachmentConfig[];
    /** Render passes: explicit depth/stencil attachment binding. */
    depthAttachment?:  DepthAttachmentConfig;

    /** Kept for format introspection / pipeline key derivation. */
    colorFormats?: GPUTextureFormat[];
    depthFormat?:  GPUTextureFormat;

    execute:       PassExecuteFn;
    hasSideEffects: boolean;
}

// -------------------------------------------------------------------------
// Compiled graph
// -------------------------------------------------------------------------

export interface CompiledGraph {
    /** Topologically sorted passes, dead passes pruned. */
    orderedPasses: PassHandle[];
    /** virtual id → physical ResourceHandle (transient or persistent). */
    resourceBindings: Map<VirtualResourceId, ResourceHandle>;
    /** Transient handles allocated this frame — released after execute(). */
    transientHandles: ResourceHandle[];
}

// -------------------------------------------------------------------------
// RenderGraph
// -------------------------------------------------------------------------

/**
 * Frame-graph implementation.
 *
 * Each frame:
 *   1. reset()        — clear state from the previous frame
 *   2. declareTexture / importTexture — register virtual resources
 *   3. addPass()      — declare passes with reads/writes and execute callbacks
 *   4. compile()      — topological sort, dead-pass culling, resource alloc
 *   5. execute()      — encode all passes into one command buffer, submit
 *
 * execute() automatically:
 *   - Creates render-pass encoders for Render passes with colorAttachments
 *   - Creates compute-pass encoders for Compute passes
 *   - Passes encoder directly (null passEncoder) for Transfer passes
 *   - Releases all transient textures back to the pool after submission
 */
export class RenderGraph {

    private _backend!: GPUBackend;
    private _resources!: ResourceManager;
    private readonly _log = new Logger('RenderGraph');

    private _nextPassHandle:  PassHandle         = 1;
    private _nextResourceId:  VirtualResourceId  = 1;

    private _passes:           Map<PassHandle, RenderPass>          = new Map();
    private _virtualResources: Map<VirtualResourceId, VirtualResource> = new Map();

    private _compiled: CompiledGraph | null = null;

    /**
     * Optional GPU timestamp query set (2 entries: frame begin + frame end).
     * When set, execute() writes timestamps at the first and last render/compute
     * pass boundaries, then resolves into the provided buffer.
     */
    private _timestampQuerySet: GPUQuerySet | null = null;
    private _timestampResolveBuffer: GPUBuffer | null = null;

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    init(backend: GPUBackend, resources: ResourceManager): void {
        this._backend   = backend;
        this._resources = resources;
    }

    /**
     * Set the timestamp query set and resolve buffer for GPU timing.
     * Pass null to disable. The query set must have count >= 2.
     * The resolve buffer must be at least 16 bytes (2 × BigUint64).
     */
    setTimestampQuery(querySet: GPUQuerySet | null, resolveBuffer: GPUBuffer | null): void {
        this._timestampQuerySet = querySet;
        this._timestampResolveBuffer = resolveBuffer;
    }

    // -------------------------------------------------------------------------
    // Frame reset
    // -------------------------------------------------------------------------

    reset(): void {
        this._passes.clear();
        this._virtualResources.clear();
        this._compiled       = null;
        this._nextPassHandle = 1;
        this._nextResourceId = 1;
    }

    // -------------------------------------------------------------------------
    // Resource declaration
    // -------------------------------------------------------------------------

    /**
     * Declare a transient texture that the graph will allocate and release.
     */
    declareTexture(name: string, desc: TextureDescriptor): VirtualResourceId {
        const id = this._nextResourceId++;
        this._virtualResources.set(id, {
            id, name, type: ResourceType.Texture, textureDesc: desc, external: false,
        });
        return id;
    }

    /**
     * Import a persistent / external texture (e.g. shadow atlas, HDR accumulation).
     * The physical handle is already known and managed outside the graph.
     */
    importTexture(name: string, physicalHandle: ResourceHandle): VirtualResourceId {
        const id = this._nextResourceId++;
        this._virtualResources.set(id, {
            id, name, type: ResourceType.Texture, physicalHandle, external: true,
        });
        return id;
    }

    // -------------------------------------------------------------------------
    // Pass declaration
    // -------------------------------------------------------------------------

    addPass(desc: {
        name:              string;
        type:              PassType;
        reads?:            VirtualResourceId[];
        writes?:           VirtualResourceId[];
        colorAttachments?: ColorAttachmentConfig[];
        depthAttachment?:  DepthAttachmentConfig;
        colorFormats?:     GPUTextureFormat[];
        depthFormat?:      GPUTextureFormat;
        hasSideEffects?:   boolean;
        execute:           PassExecuteFn;
    }): PassHandle {
        const handle = this._nextPassHandle++;

        // Derive writes from explicit attachments if not supplied separately
        const attachmentWrites: VirtualResourceId[] = [];
        for (const ca of desc.colorAttachments ?? []) {
            if (ca.storeOp === 'store') attachmentWrites.push(ca.resourceId);
        }
        if (desc.depthAttachment?.depthStoreOp === 'store' && !desc.depthAttachment.depthReadOnly) {
            attachmentWrites.push(desc.depthAttachment.resourceId);
        }

        const writes = desc.writes ?? attachmentWrites;

        // Also collect read-only attachment uses into reads
        const attachmentReads: VirtualResourceId[] = [];
        for (const ca of desc.colorAttachments ?? []) {
            if (ca.loadOp === 'load') attachmentReads.push(ca.resourceId);
        }
        if (desc.depthAttachment?.depthLoadOp === 'load' && !desc.depthAttachment.depthReadOnly) {
            attachmentReads.push(desc.depthAttachment.resourceId);
        }

        const reads = [...(desc.reads ?? []), ...attachmentReads];

        this._passes.set(handle, {
            handle,
            name:             desc.name,
            type:             desc.type,
            reads,
            writes,
            colorAttachments: desc.colorAttachments,
            depthAttachment:  desc.depthAttachment,
            colorFormats:     desc.colorFormats,
            depthFormat:      desc.depthFormat,
            execute:          desc.execute,
            hasSideEffects:   desc.hasSideEffects ?? false,
        });
        return handle;
    }

    // -------------------------------------------------------------------------
    // Well-known pass helpers
    // -------------------------------------------------------------------------

    addGBufferPass(
        colorAttachments: ColorAttachmentConfig[],
        depthAttachment:  DepthAttachmentConfig,
        execute: PassExecuteFn,
    ): PassHandle {
        return this.addPass({
            name:             'GBuffer',
            type:             PassType.Render,
            colorAttachments,
            depthAttachment,
            colorFormats:     ['rgba8unorm', 'rgba16float', 'rgba8unorm'],
            depthFormat:      'depth32float',
            hasSideEffects:   false,
            execute,
        });
    }

    addLightingPass(
        reads:            VirtualResourceId[],
        colorAttachments: ColorAttachmentConfig[],
        execute:          PassExecuteFn,
        asCompute = false,
    ): PassHandle {
        return this.addPass({
            name:             'DeferredLighting',
            type:             asCompute ? PassType.Compute : PassType.Render,
            reads,
            colorAttachments: asCompute ? undefined : colorAttachments,
            hasSideEffects:   false,
            execute,
        });
    }

    addForwardPass(
        reads:            VirtualResourceId[],
        colorAttachments: ColorAttachmentConfig[],
        depthAttachment:  DepthAttachmentConfig,
        execute:          PassExecuteFn,
    ): PassHandle {
        return this.addPass({
            name:             'ForwardTransparent',
            type:             PassType.Render,
            reads,
            colorAttachments,
            depthAttachment,
            colorFormats:     ['rgba16float'],
            depthFormat:      'depth32float',
            hasSideEffects:   false,
            execute,
        });
    }

    /**
     * Depth-only prepass — no colour attachments.
     *
     * Clears and fills the shared depth buffer before the G-Buffer pass.
     * The G-Buffer pass should subsequently be declared with
     * `depthLoadOp: 'load'` and a pipeline that uses `depthCompare: 'equal'`
     * so the GPU early-Z unit rejects occluded fragments without invoking
     * the fragment shader.
     */
    addDepthPrepassPass(
        depthAttachment: DepthAttachmentConfig,
        execute: PassExecuteFn,
    ): PassHandle {
        return this.addPass({
            name:           'DepthPrepass',
            type:           PassType.Render,
            depthAttachment,
            depthFormat:    'depth32float',
            hasSideEffects: false,
            execute,
        });
    }

    addShadowPass(
        depthAttachment: DepthAttachmentConfig,
        execute: PassExecuteFn,
    ): PassHandle {
        return this.addPass({
            name:           'ShadowMap',
            type:           PassType.Render,
            depthAttachment,
            depthFormat:    'depth32float',
            // The shadow atlas is a persistent external texture not tracked as a
            // virtual resource read by the lighting pass. Mark as side-effects so
            // the pass is never culled by dead-pass elimination.
            hasSideEffects: true,
            execute,
        });
    }

    /** Final blit / post-process to the swap-chain (always has side effects). */
    addPresentPass(
        reads:            VirtualResourceId[],
        colorAttachments: ColorAttachmentConfig[],
        execute:          PassExecuteFn,
    ): PassHandle {
        return this.addPass({
            name:             'Present',
            type:             PassType.Render,
            reads,
            colorAttachments,
            hasSideEffects:   true,   // always keep — final output
            execute,
        });
    }

    addRTPass(
        reads: VirtualResourceId[], writes: VirtualResourceId[], execute: PassExecuteFn,
    ): PassHandle {
        return this.addPass({
            name: 'RayTrace', type: PassType.RayTrace,
            reads, writes, hasSideEffects: false, execute,
        });
    }

    // -------------------------------------------------------------------------
    // Compilation
    // -------------------------------------------------------------------------

    /**
     * Compile the graph:
     *   1. Topological sort via Kahn's algorithm
     *   2. Backward reachability from side-effect passes → prune dead passes
     *   3. Allocate transient GPU textures for virtual resources without physicalHandle
     */
    compile(): CompiledGraph {
        const passes = [...this._passes.values()];

        // ---- Step 1: topological sort (Kahn's) ------------------------------
        // resource id → pass handles that WRITE it
        const resourceWriters = new Map<VirtualResourceId, Set<PassHandle>>();
        for (const pass of passes) {
            for (const vid of pass.writes) {
                let s = resourceWriters.get(vid);
                if (!s) { s = new Set(); resourceWriters.set(vid, s); }
                s.add(pass.handle);
            }
        }

        // Build edges: A → B if B reads a resource written by A
        const outEdges = new Map<PassHandle, Set<PassHandle>>();
        const inDegree = new Map<PassHandle, number>();
        for (const pass of passes) {
            outEdges.set(pass.handle, new Set());
            inDegree.set(pass.handle, 0);
        }
        for (const pass of passes) {
            for (const vid of pass.reads) {
                const writers = resourceWriters.get(vid);
                if (!writers) continue;
                for (const writerHandle of writers) {
                    if (writerHandle === pass.handle) continue;
                    // Only increment inDegree when the edge is genuinely NEW.
                    // outEdges uses a Set, so duplicate resource reads that map
                    // to the same writer (e.g. all G-Buffer textures → GBuffer pass)
                    // must not be double-counted.
                    const edgeSet = outEdges.get(writerHandle)!;
                    if (!edgeSet.has(pass.handle)) {
                        edgeSet.add(pass.handle);
                        inDegree.set(pass.handle, (inDegree.get(pass.handle) ?? 0) + 1);
                    }
                }
            }
        }

        // Kahn BFS
        const queue: PassHandle[] = [];
        for (const [handle, deg] of inDegree) {
            if (deg === 0) queue.push(handle);
        }

        const sorted: PassHandle[] = [];
        while (queue.length > 0) {
            const handle = queue.shift()!;
            sorted.push(handle);
            for (const successor of outEdges.get(handle) ?? []) {
                const newDeg = (inDegree.get(successor) ?? 1) - 1;
                inDegree.set(successor, newDeg);
                if (newDeg === 0) queue.push(successor);
            }
        }

        if (sorted.length !== passes.length) {
            this._log.warn(`RenderGraph: cycle detected — ${passes.length - sorted.length} pass(es) dropped`);
        }

        // ---- Step 2: backward reachability from sink (hasSideEffects) passes --
        const reachable = new Set<PassHandle>();
        const sinkQueue: PassHandle[] = sorted.filter(
            h => this._passes.get(h)!.hasSideEffects,
        );

        // Build reverse edges for backward walk
        const inEdges = new Map<PassHandle, Set<PassHandle>>();
        for (const pass of passes) inEdges.set(pass.handle, new Set());
        for (const [from, toSet] of outEdges) {
            for (const to of toSet) inEdges.get(to)!.add(from);
        }

        const bfsQueue = [...sinkQueue];
        while (bfsQueue.length > 0) {
            const h = bfsQueue.shift()!;
            if (reachable.has(h)) continue;
            reachable.add(h);
            for (const dep of inEdges.get(h) ?? []) bfsQueue.push(dep);
        }

        const orderedPasses = sorted.filter(h => reachable.has(h));

        // ---- Step 3: allocate transient resources ---------------------------
        const resourceBindings = new Map<VirtualResourceId, ResourceHandle>();
        const transientHandles: ResourceHandle[] = [];

        for (const vr of this._virtualResources.values()) {
            if (vr.physicalHandle !== undefined) {
                // external / already-allocated
                resourceBindings.set(vr.id, vr.physicalHandle);
            } else if (vr.textureDesc) {
                const rh = this._resources.acquireTransientTexture(vr.textureDesc);
                vr.physicalHandle = rh;
                resourceBindings.set(vr.id, rh);
                transientHandles.push(rh);
            }
        }

        this._log.debug(
            `Compiled: ${orderedPasses.length}/${passes.length} passes kept, ` +
            `${transientHandles.length} transient textures allocated`
        );

        this._compiled = { orderedPasses, resourceBindings, transientHandles };
        return this._compiled;
    }

    // -------------------------------------------------------------------------
    // Execution
    // -------------------------------------------------------------------------

    /**
     * Walk compiled passes, create one GPUCommandEncoder, begin/end per-pass
     * encoders automatically, invoke each execute callback, then submit.
     * Transient textures are released back to the pool after submission.
     */
    execute(): void {
        if (!this._compiled) {
            throw new Error('[RenderGraph] execute() called before compile()');
        }

        const encoder = this._backend.device.createCommandEncoder({ label: 'RenderGraph' });
        const tsQS = this._timestampQuerySet;

        // Find the first and last render/compute pass indices for timestamp injection.
        const ordered = this._compiled.orderedPasses;
        let firstPassIdx = -1;
        let lastPassIdx  = -1;
        if (tsQS) {
            for (let i = 0; i < ordered.length; i++) {
                const p = this._passes.get(ordered[i]!);
                if (p && (p.type === PassType.Render || p.type === PassType.Compute)) {
                    if (firstPassIdx === -1) firstPassIdx = i;
                    lastPassIdx = i;
                }
            }
        }

        for (let i = 0; i < ordered.length; i++) {
            const pass = this._passes.get(ordered[i]!);
            if (!pass) continue;

            // Build timestampWrites for the first and/or last render/compute pass.
            let timestampWrites: GPURenderPassTimestampWrites | undefined;
            if (tsQS && (i === firstPassIdx || i === lastPassIdx) &&
                (pass.type === PassType.Render || pass.type === PassType.Compute)) {
                timestampWrites = { querySet: tsQS };
                if (i === firstPassIdx) (timestampWrites as any).beginningOfPassWriteIndex = 0;
                if (i === lastPassIdx)  (timestampWrites as any).endOfPassWriteIndex = 1;
            }

            switch (pass.type) {
                case PassType.Render:
                    this._executeRenderPass(encoder, pass, timestampWrites);
                    break;

                case PassType.Compute: {
                    const computeEncoder = encoder.beginComputePass({
                        label: pass.name,
                        timestampWrites: timestampWrites as GPUComputePassTimestampWrites | undefined,
                    });
                    pass.execute(encoder, computeEncoder);
                    computeEncoder.end();
                    break;
                }

                case PassType.Transfer:
                case PassType.RayTrace:
                    pass.execute(encoder, null);
                    break;
            }
        }

        // Resolve timestamp queries into the readback buffer.
        if (tsQS && this._timestampResolveBuffer && firstPassIdx !== -1) {
            encoder.resolveQuerySet(tsQS, 0, 2, this._timestampResolveBuffer, 0);
        }

        this._backend.queue.submit([encoder.finish()]);

        // Release transient textures back to pool
        for (const rh of this._compiled.transientHandles) {
            this._resources.releaseTransientTexture(rh);
        }
    }

    // -------------------------------------------------------------------------
    // Public accessors
    // -------------------------------------------------------------------------

    /** Resolve a virtual resource id to its physical ResourceHandle. */
    getPhysicalHandle(id: VirtualResourceId): ResourceHandle | undefined {
        return this._compiled?.resourceBindings.get(id);
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    private _executeRenderPass(encoder: GPUCommandEncoder, pass: RenderPass, timestampWrites?: GPURenderPassTimestampWrites): void {
        // If the pass has no explicit attachment config, delegate fully to the callback
        if (!pass.colorAttachments && !pass.depthAttachment) {
            pass.execute(encoder, null);
            return;
        }

        // --- Build color attachments -----------------------------------------
        const colorAttachments: GPURenderPassColorAttachment[] = [];
        for (const ca of pass.colorAttachments ?? []) {
            const view = this._resolveView(ca.resourceId);
            if (!view) {
                this._log.warn(`Pass "${pass.name}": color attachment ${ca.resourceId} has no physical resource — skipping`);
                continue;
            }

            const attachment: GPURenderPassColorAttachment = {
                view,
                loadOp:     ca.loadOp,
                storeOp:    ca.storeOp,
                clearValue: ca.clearColor ?? { r: 0, g: 0, b: 0, a: 1 },
            };

            if (ca.resolveTargetId !== undefined) {
                const resolveView = this._resolveView(ca.resolveTargetId);
                if (resolveView) attachment.resolveTarget = resolveView;
            }

            colorAttachments.push(attachment);
        }

        // --- Build depth-stencil attachment ----------------------------------
        let depthStencilAttachment: GPURenderPassDepthStencilAttachment | undefined;
        if (pass.depthAttachment) {
            const da   = pass.depthAttachment;
            const view = this._resolveView(da.resourceId, { aspect: 'depth-only' });
            if (view) {
                if (da.depthReadOnly) {
                    depthStencilAttachment = {
                        view,
                        depthReadOnly: true,
                    };
                } else {
                    depthStencilAttachment = {
                        view,
                        depthLoadOp:    da.depthLoadOp,
                        depthStoreOp:   da.depthStoreOp,
                        depthClearValue: da.depthClearValue ?? 1.0,
                    };
                }
                if (da.stencilLoadOp !== undefined) {
                    depthStencilAttachment.stencilLoadOp  = da.stencilLoadOp;
                    depthStencilAttachment.stencilStoreOp = da.stencilStoreOp ?? 'discard';
                    depthStencilAttachment.stencilClearValue = da.stencilClearValue ?? 0;
                }
            }
        }

        const renderPassEncoder = encoder.beginRenderPass({
            label:                  pass.name,
            colorAttachments,
            depthStencilAttachment,
            timestampWrites,
        });

        pass.execute(encoder, renderPassEncoder);
        renderPassEncoder.end();
    }

    /**
     * Resolve a VirtualResourceId to a GPUTextureView using the compiled
     * resource bindings.  Returns undefined if the resource is unresolved.
     */
    private _resolveView(
        id: VirtualResourceId,
        viewDesc?: GPUTextureViewDescriptor,
    ): GPUTextureView | undefined {
        const handle = this._compiled?.resourceBindings.get(id);
        if (handle === undefined) return undefined;
        return this._resources.createView(handle, viewDesc);
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    destroy(): void {
        this.reset();
    }
}
