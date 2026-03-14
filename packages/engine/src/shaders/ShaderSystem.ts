// /src/engine/shaders/ShaderSystem.ts

import { Logger } from '../core/Logger';
import type { GPUBackend } from '../core/GPUBackend';

// -------------------------------------------------------------------------
// Public types
// -------------------------------------------------------------------------

export type ShaderHandle = number;

/**
 * Key-value defines injected into WGSL source before compilation.
 *
 *  Feature flags  → { "HAS_NORMAL_MAP": "1" }           used with #ifdef
 *  Value injection → { "MAX_LIGHTS": "256" }             used as #{MAX_LIGHTS}
 */
export type ShaderDefines = Record<string, string>;

export interface ShaderSourceDescriptor {
    label: string;
    /** Raw WGSL source, may contain preprocessor directives. */
    source: string;
}

/** A compiled variant identified by base shader label + its defines. */
export interface ShaderVariant {
    handle: ShaderHandle;
    module: GPUShaderModule;
    defines: ShaderDefines;
    /** The fully-preprocessed WGSL sent to the driver. */
    processedSource: string;
}

/** Bind-group slot parsed from WGSL source annotations. */
export interface BindingInfo {
    group:   number;
    binding: number;
    name:    string;
    /** 'uniform' | 'storage' | 'texture' | 'sampler' | 'storageTexture' */
    kind: string;
    /** Original WGSL type string, e.g. "texture_2d<f32>" */
    wgslType: string;
    access?: string; // 'read' | 'read_write' (for storage buffers)
}

/** Vertex input parsed from WGSL @location annotations. */
export interface VertexInputInfo {
    location: number;
    name: string;
    wgslType: string;
}

export interface ShaderReflection {
    bindings: BindingInfo[];
    vertexInputs: VertexInputInfo[];
    /** Same bindings organized as bind-group-layout entry arrays. */
    bindGroupLayouts: GPUBindGroupLayoutEntry[][];
}

// -------------------------------------------------------------------------
// ShaderSystem
// -------------------------------------------------------------------------

/** Maximum nested #include depth before we throw to prevent infinite recursion. */
const MAX_INCLUDE_DEPTH = 16;

/**
 * Manages WGSL shader loading, preprocessing, variant generation,
 * caching of GPUShaderModules, and basic reflection.
 *
 * Preprocessor directives supported in WGSL source files:
 *
 *   #include "snippet-name"          — insert registered include snippet
 *   #ifdef  DEFINE_NAME              — emit block if define is present
 *   #ifndef DEFINE_NAME              — emit block if define is absent
 *   #else                            — flip current block
 *   #endif                           — close conditional block
 *   #{DEFINE_NAME}                   — inline value substitution
 *
 * Usage:
 *   shaders.registerSource({ label: 'gbuffer.vert', source: wgslSrc });
 *   shaders.registerInclude('common', commonWgsl);
 *   const variant = shaders.getVariant('gbuffer.vert', { HAS_NORMAL_MAP: '1' });
 *   // variant.module is ready for use in a pipeline descriptor
 */
export class ShaderSystem {

    private _backend!: GPUBackend;
    private _nextHandle: ShaderHandle = 1;
    private readonly _log = new Logger('ShaderSystem');

    /** label → raw WGSL source */
    private _sources: Map<string, string> = new Map();

    /** variant cache key → compiled variant */
    private _variantCache: Map<string, ShaderVariant> = new Map();

    /** handle → compiled variant  (for PipelineManager handle resolution) */
    private _handleToVariant: Map<ShaderHandle, ShaderVariant> = new Map();

    /** include name → WGSL snippet source */
    private _includeLibrary: Map<string, string> = new Map();

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    init(backend: GPUBackend): void {
        this._backend = backend;
    }

    // -------------------------------------------------------------------------
    // Source registration
    // -------------------------------------------------------------------------

    registerSource(desc: ShaderSourceDescriptor): void {
        this._sources.set(desc.label, desc.source);
        this._log.debug(`Registered shader source "${desc.label}" (${desc.source.length} chars)`);
    }

    registerInclude(name: string, source: string): void {
        this._includeLibrary.set(name, source);
        this._log.debug(`Registered include "${name}"`);
    }

    /** Fetch a .wgsl file by URL and register it under the given label. */
    async loadFromURL(label: string, url: string): Promise<void> {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`[ShaderSystem] Failed to load "${label}" from ${url}: ${response.status} ${response.statusText}`);
        }
        const source = await response.text();
        this.registerSource({ label, source });
    }

    /** True if a source has been registered under this label. */
    hasSource(label: string): boolean {
        return this._sources.has(label);
    }

    // -------------------------------------------------------------------------
    // Preprocessing
    // -------------------------------------------------------------------------

    /**
     * Run the full preprocessor pipeline on raw WGSL source:
     *   1. Resolve #include directives (recursive, depth-limited)
     *   2. Evaluate #ifdef / #ifndef / #else / #endif conditional blocks
     *   3. Substitute #{KEY} value tokens with their define values
     *
     * Returns the final WGSL string ready for device.createShaderModule.
     */
    preprocess(source: string, defines: ShaderDefines, _depth = 0): string {
        if (_depth > MAX_INCLUDE_DEPTH) {
            throw new Error('[ShaderSystem] Maximum #include depth exceeded — possible circular include');
        }

        const lines  = source.split('\n');
        const output: string[] = [];

        // Condition stack: each entry reflects a nesting level.
        // emitting = whether lines at this level should be written to output.
        const stack: { emitting: boolean; seenElse: boolean }[] = [
            { emitting: true, seenElse: false },
        ];

        const isEmitting = (): boolean => stack.every(s => s.emitting);

        for (const rawLine of lines) {
            const trimmed = rawLine.trimStart();

            // ---- #include "name" -------------------------------------------
            const includeMatch = trimmed.match(/^#include\s+"([^"]+)"/);
            if (includeMatch) {
                if (isEmitting()) {
                    const name    = includeMatch[1]!;
                    const snippet = this._includeLibrary.get(name);
                    if (snippet === undefined) {
                        throw new Error(`[ShaderSystem] #include "${name}" not found in include library`);
                    }
                    const expanded = this.preprocess(snippet, defines, _depth + 1);
                    output.push(expanded);
                }
                continue;
            }

            // ---- #ifdef DEFINE ----------------------------------------------
            const ifdefMatch = trimmed.match(/^#ifdef\s+(\w+)/);
            if (ifdefMatch) {
                const defined = Object.prototype.hasOwnProperty.call(defines, ifdefMatch[1]!);
                stack.push({ emitting: isEmitting() && defined, seenElse: false });
                continue;
            }

            // ---- #ifndef DEFINE ---------------------------------------------
            const ifndefMatch = trimmed.match(/^#ifndef\s+(\w+)/);
            if (ifndefMatch) {
                const defined = Object.prototype.hasOwnProperty.call(defines, ifndefMatch[1]!);
                stack.push({ emitting: isEmitting() && !defined, seenElse: false });
                continue;
            }

            // ---- #else ------------------------------------------------------
            if (trimmed.startsWith('#else')) {
                const top = stack[stack.length - 1]!;
                if (top.seenElse) throw new Error('[ShaderSystem] Unexpected #else after #else');
                // Flip emitting only if the parent level is emitting
                const parentEmitting = stack.slice(0, -1).every(s => s.emitting);
                top.emitting = parentEmitting && !top.emitting;
                top.seenElse = true;
                continue;
            }

            // ---- #endif -----------------------------------------------------
            if (trimmed.startsWith('#endif')) {
                if (stack.length <= 1) throw new Error('[ShaderSystem] Unexpected #endif without matching #ifdef');
                stack.pop();
                continue;
            }

            // ---- regular line -----------------------------------------------
            if (isEmitting()) {
                output.push(rawLine);
            }
        }

        if (stack.length > 1) {
            throw new Error(`[ShaderSystem] ${stack.length - 1} unclosed #ifdef/#ifndef block(s)`);
        }

        // ---- value substitution: #{KEY} → defines[KEY] ---------------------
        let result = output.join('\n');
        result = result.replace(/#\{(\w+)\}/g, (_, key: string) => {
            if (Object.prototype.hasOwnProperty.call(defines, key)) {
                return defines[key]!;
            }
            this._log.warn(`#{${key}} used in shader but not present in defines — left as empty string`);
            return '';
        });

        return result;
    }

    // -------------------------------------------------------------------------
    // Compilation & variant management
    // -------------------------------------------------------------------------

    /**
     * Return a cached or newly compiled shader variant.
     * Compilation errors are logged asynchronously via getCompilationInfo().
     */
    getVariant(label: string, defines: ShaderDefines = {}): ShaderVariant {
        const key    = this._variantKey(label, defines);
        const cached = this._variantCache.get(key);
        if (cached) return cached;

        const rawSource = this._sources.get(label);
        if (rawSource === undefined) {
            throw new Error(`[ShaderSystem] No source registered for "${label}"`);
        }

        const processedSource = this.preprocess(rawSource, defines);

        const module = this._backend.device.createShaderModule({
            label:  `${label}${_definesLabel(defines)}`,
            code:   processedSource,
        });

        // Async compilation error reporting (non-blocking)
        module.getCompilationInfo().then(info => {
            for (const msg of info.messages) {
                const loc = `${msg.lineNum}:${msg.linePos}`;
                if (msg.type === 'error') {
                    this._log.error(`Shader "${label}" compile error at ${loc}: ${msg.message}`);
                } else if (msg.type === 'warning') {
                    this._log.warn(`Shader "${label}" warning at ${loc}: ${msg.message}`);
                }
            }
        });

        const variant: ShaderVariant = {
            handle: this._nextHandle++,
            module,
            defines,
            processedSource,
        };

        this._variantCache.set(key, variant);
        this._handleToVariant.set(variant.handle, variant);
        this._log.info(`Compiled shader "${label}" handle=${variant.handle} defines=${JSON.stringify(defines)}`);
        return variant;
    }

    /**
     * Resolve a ShaderHandle back to its compiled variant.
     * Used by PipelineManager to obtain GPUShaderModule from a stored handle.
     */
    getVariantByHandle(handle: ShaderHandle): ShaderVariant | undefined {
        return this._handleToVariant.get(handle);
    }

    /**
     * Force-compile multiple define permutations up front (warm cache).
     * Safe to call at load time to avoid first-frame stalls.
     */
    precompileVariants(label: string, definesSets: ShaderDefines[]): void {
        for (const defines of definesSets) {
            this.getVariant(label, defines);
        }
    }

    // -------------------------------------------------------------------------
    // Reflection
    // -------------------------------------------------------------------------

    /**
     * Parse the preprocessed WGSL source to extract:
     *  - All @group / @binding declarations with their type and access mode
     *  - All @location vertex inputs
     *  - A ready-to-use GPUBindGroupLayoutEntry[][] derived from the above
     *
     * This is regex-based, not a full WGSL AST parser. It relies on the
     * conventional annotation style used throughout this engine's shaders.
     */
    reflect(variant: ShaderVariant): ShaderReflection {
        const src = variant.processedSource;

        // ---- bindings -------------------------------------------------------
        const bindings = _parseBindings(src);

        // ---- vertex inputs ---------------------------------------------------
        const vertexInputs = _parseVertexInputs(src);

        // ---- build GPUBindGroupLayoutEntry[][] ------------------------------
        const maxGroup = bindings.reduce((m, b) => Math.max(m, b.group), -1);
        const bindGroupLayouts: GPUBindGroupLayoutEntry[][] = [];

        for (let g = 0; g <= maxGroup; g++) {
            const groupBindings = bindings.filter(b => b.group === g);
            bindGroupLayouts.push(groupBindings.map(b => _bindingToLayoutEntry(b)));
        }

        return { bindings, vertexInputs, bindGroupLayouts };
    }

    // -------------------------------------------------------------------------
    // Cache management
    // -------------------------------------------------------------------------

    clearCache(): void {
        this._variantCache.clear();
        this._handleToVariant.clear();
        this._log.debug('Variant cache cleared');
    }

    destroy(): void {
        this._sources.clear();
        this._includeLibrary.clear();
        this._variantCache.clear();
        this._handleToVariant.clear();
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    private _variantKey(label: string, defines: ShaderDefines): string {
        const sorted = Object.entries(defines).sort(([a], [b]) => a.localeCompare(b));
        return `${label}|${JSON.stringify(sorted)}`;
    }
}

// -------------------------------------------------------------------------
// Module-level helpers
// -------------------------------------------------------------------------

function _definesLabel(defines: ShaderDefines): string {
    const keys = Object.keys(defines);
    if (keys.length === 0) return '';
    return `[${keys.sort().join(',')}]`;
}

// ---- Reflection helpers ----------------------------------------------------

/**
 * Parse all @group/@binding declarations from WGSL source.
 *
 * Matches patterns like:
 *   @group(0) @binding(1) var<uniform> camera : CameraUniforms;
 *   @group(1) @binding(0) var albedo : texture_2d<f32>;
 *   @group(1) @binding(2) var<storage, read_write> particles : array<Particle>;
 */
function _parseBindings(src: string): BindingInfo[] {
    const results: BindingInfo[] = [];

    // One regex to capture group, binding, var kind (<...>), name, and WGSL type
    const re = /@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<([^>]*)>)?\s+(\w+)\s*:\s*([^;]+?)\s*;/g;
    let m: RegExpExecArray | null;

    while ((m = re.exec(src)) !== null) {
        const [, groupStr, bindingStr, varKindRaw, name, typeRaw] = m;
        const group   = parseInt(groupStr!,   10);
        const binding = parseInt(bindingStr!, 10);
        const varKind = varKindRaw?.trim() ?? '';
        const wgslType = typeRaw!.trim();

        // Determine kind and access from the var<...> qualifier
        let kind   = 'uniform';
        let access: string | undefined;

        if (varKind.startsWith('storage')) {
            kind = 'storage';
            access = varKind.includes('read_write') ? 'read_write' : 'read';
        } else if (varKind === '' || varKind === 'handle') {
            // Texture or sampler — determine from the WGSL type
            if (wgslType.startsWith('texture_storage')) {
                kind = 'storageTexture';
            } else if (wgslType.startsWith('texture')) {
                kind = 'texture';
            } else if (wgslType.startsWith('sampler')) {
                kind = 'sampler';
            }
        } else if (varKind === 'uniform') {
            kind = 'uniform';
        }

        results.push({ group, binding, name: name!, kind, wgslType, access });
    }

    return results;
}

/**
 * Parse @location vertex inputs from a WGSL vertex entry-point struct.
 *
 * Matches:
 *   @location(0) position : vec4<f32>,
 *   @location(1) normal   : vec3<f32>,
 */
function _parseVertexInputs(src: string): VertexInputInfo[] {
    const results: VertexInputInfo[] = [];
    const re = /@location\((\d+)\)\s+(\w+)\s*:\s*([\w<>,\s]+?)(?=[,;\)])/g;
    let m: RegExpExecArray | null;

    while ((m = re.exec(src)) !== null) {
        const [, locStr, name, typeRaw] = m;
        results.push({
            location: parseInt(locStr!, 10),
            name:     name!,
            wgslType: typeRaw!.trim(),
        });
    }

    return results;
}

/**
 * Convert a parsed BindingInfo into a GPUBindGroupLayoutEntry.
 * Uses the most permissive visibility (VERTEX | FRAGMENT | COMPUTE)
 * so layouts are shareable across passes. Individual passes can restrict if needed.
 */
function _bindingToLayoutEntry(b: BindingInfo): GPUBindGroupLayoutEntry {
    const visibility = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE;
    const entry: GPUBindGroupLayoutEntry = { binding: b.binding, visibility };

    switch (b.kind) {
        case 'uniform':
            entry.buffer = { type: 'uniform' };
            break;

        case 'storage':
            entry.buffer = { type: b.access === 'read_write' ? 'storage' : 'read-only-storage' };
            break;

        case 'texture': {
            // Infer sampleType from WGSL type string
            const sampleType = b.wgslType.includes('<f32>') ? 'float'
                             : b.wgslType.includes('<i32>') ? 'sint'
                             : b.wgslType.includes('<u32>') ? 'uint'
                             : b.wgslType.includes('depth') ? 'depth'
                             : 'float';
            const multisampled = b.wgslType.includes('multisampled');
            const viewDimension: GPUTextureViewDimension =
                b.wgslType.includes('cube_array')   ? 'cube-array'
              : b.wgslType.includes('cube')          ? 'cube'
              : b.wgslType.includes('_2d_array')     ? '2d-array'
              : b.wgslType.includes('_3d')            ? '3d'
              : '2d';
            entry.texture = { sampleType: sampleType as GPUTextureSampleType, multisampled, viewDimension };
            break;
        }

        case 'storageTexture': {
            // e.g. texture_storage_2d<rgba8unorm, write>
            const formatMatch = b.wgslType.match(/texture_storage_\w+<([\w]+)/);
            const accessMatch = b.wgslType.match(/,\s*(read|write|read_write)/);
            const format      = (formatMatch?.[1] ?? 'rgba8unorm') as GPUTextureFormat;
            const access      = (accessMatch?.[1] === 'read_write' ? 'read-write'
                               : accessMatch?.[1] === 'read'       ? 'read-only'
                               : 'write-only') as GPUStorageTextureAccess;
            entry.storageTexture = { format, access, viewDimension: '2d' };
            break;
        }

        case 'sampler':
            entry.sampler = {
                type: b.wgslType.includes('comparison') ? 'comparison' : 'filtering',
            };
            break;
    }

    return entry;
}
