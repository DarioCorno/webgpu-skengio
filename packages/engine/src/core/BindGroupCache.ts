// /src/engine/core/BindGroupCache.ts
//
// Lightweight cache for a single GPUBindGroup.
// Avoids redundant per-frame createBindGroup() calls when the underlying
// physical resources (GPUTextureView, GPUBuffer, GPUSampler) haven't changed.
// Uses reference equality — works because ResourceManager.createView() already
// caches views per (handle, viewDesc), so same handle → same object reference.

/**
 * Caches a single GPUBindGroup, invalidated when any bound resource
 * changes identity (reference equality).
 */
export class BindGroupCache {
    private _bg: GPUBindGroup | null = null;
    private _keys: unknown[] = [];

    /**
     * Returns the cached bind group if all identity keys still match,
     * otherwise invokes `factory`, caches the result, and returns it.
     *
     * @param keys    Resource references to compare (GPUTextureView, GPUBuffer, GPUSampler, etc.)
     * @param factory Called on cache miss to create the new bind group.
     */
    getOrCreate(keys: unknown[], factory: () => GPUBindGroup): GPUBindGroup {
        const prev = this._keys;
        if (this._bg !== null && keys.length === prev.length) {
            let match = true;
            for (let i = 0; i < keys.length; i++) {
                if (keys[i] !== prev[i]) { match = false; break; }
            }
            if (match) return this._bg;
        }
        this._bg = factory();
        this._keys = keys;
        return this._bg;
    }

    clear(): void {
        this._bg = null;
        this._keys.length = 0;
    }
}
