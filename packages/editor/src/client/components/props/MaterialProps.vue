<script setup lang="ts">
import { computed, ref, inject } from 'vue';
import type { Ref } from 'vue';
import { type Engine, type MaterialRecord } from '@skengio/engine';
import NumericInput from '../NumericInput.vue';
import TexturePicker from '../TexturePicker.vue';
import ColorSwatch from '../ColorSwatch.vue';
import { useEngineTick } from '../../composables/useEngineTick';
import { useHintPanel } from '../../composables/useHintPanel';
import { materialHints } from '../../hints/materialHints';

const props = defineProps<{
    engine: Engine;
    materialHandle: number;
}>();

const { tick, bump } = useEngineTick();
const { hintEvents } = useHintPanel();
const hint = (key: string) => hintEvents(materialHints[key]!);

const mat = computed(() => {
    tick.value;
    const m = props.engine.materials.getMaterial(props.materialHandle);
    return m ? { ...m } : undefined;
});

function updatePBR(field: string, value: number) {
    const m = mat.value;
    if (!m) return;
    props.engine.materials.updateMaterial(props.materialHandle, {
        pbrParams: { ...m.pbrParams, [field]: value },
    });
    bump();
}

function updateBaseColor(axis: number, value: number) {
    const m = mat.value;
    if (!m) return;
    const c: [number, number, number, number] = [...m.pbrParams.baseColorFactor];
    c[axis] = Math.max(0, Math.min(1, value));
    props.engine.materials.updateMaterial(props.materialHandle, {
        pbrParams: { ...m.pbrParams, baseColorFactor: c },
    });
    bump();
}

function updateEmissive(axis: number, value: number) {
    const m = mat.value;
    if (!m) return;
    const e: [number, number, number] = [...m.pbrParams.emissiveFactor];
    e[axis] = Math.max(0, value);
    props.engine.materials.updateMaterial(props.materialHandle, {
        pbrParams: { ...m.pbrParams, emissiveFactor: e },
    });
    bump();
}

function toggleDoubleSided() {
    const m = mat.value;
    if (!m) return;
    props.engine.materials.updateMaterial(props.materialHandle, {
        doubleSided: !m.doubleSided,
    });
    bump();
}

function toggleCastShadow() {
    const m = mat.value;
    if (!m) return;
    props.engine.materials.updateMaterial(props.materialHandle, {
        castShadow: !m.castShadow,
    });
    bump();
}

function hasTexture(field: string): boolean {
    const m = mat.value;
    return m ? (m.textures as any)[field] !== undefined : false;
}

function onSwatchUpdate(r: number, g: number, b: number) {
    const m = mat.value;
    if (!m) return;
    const c: [number, number, number, number] = [r, g, b, m.pbrParams.baseColorFactor[3]];
    props.engine.materials.updateMaterial(props.materialHandle, {
        pbrParams: { ...m.pbrParams, baseColorFactor: c },
    });
    bump();
}

// Texture slots
const pickerSlot = ref<string | null>(null);

// Scene JSON data — texture names and material texture refs live here,
// surviving component re-mounts across node switches.
const sceneData = inject<Ref<Record<string, any>>>('sceneData', ref({}));

const texSlots: { field: string; label: string }[] = [
    { field: 'baseColorMap', label: 'Base Color' },
    { field: 'normalMap', label: 'Normal' },
    { field: 'metallicRoughnessMap', label: 'Metal/Rough' },
    { field: 'occlusionMap', label: 'AO' },
    { field: 'emissiveMap', label: 'Emissive' },
];

// The material label matches the key in sceneData.materials
function getSceneMat(): Record<string, any> | undefined {
    const label = mat.value?.label;
    if (!label) return undefined;
    return sceneData.value?.materials?.[label];
}

// Resolve a texture slot to a URL for preview.
// Scene material stores a texture *key* (e.g. "rust_diffuse"),
// and sceneData.textures maps keys to URLs (e.g. "/textures/rust-ground-diffuse.jpg").
// For textures added via the editor (not in the original scene), the key IS the file path.
function texPreviewUrl(field: string): string | null {
    const sceneMat = getSceneMat();
    const texKey = sceneMat?.[field];
    if (!texKey) return null;
    // Look up in scene textures table first
    const url = sceneData.value?.textures?.[texKey];
    if (url) return url;
    // If key is already a path (editor-added), use it directly
    return `/textures/${texKey}`;
}

function texDisplayName(field: string): string {
    const sceneMat = getSceneMat();
    const texKey = sceneMat?.[field];
    if (!texKey) return '';
    // Show the URL filename from the textures table, or the key itself
    const url = sceneData.value?.textures?.[texKey] ?? texKey;
    const parts = url.split('/');
    return parts[parts.length - 1];
}

function openPicker(field: string) {
    pickerSlot.value = field;
}

function slotLabel(field: string): string {
    return texSlots.find(s => s.field === field)?.label ?? field;
}

// Write a texture reference into the scene data and ensure a textures entry exists.
function setSceneTexture(field: string, file: string) {
    const sd = sceneData.value;
    if (!sd) return;
    // Ensure textures table exists
    if (!sd.textures) sd.textures = {};
    // Use the file path as the texture key (editor-added textures)
    const texKey = file;
    sd.textures[texKey] = `/textures/${file}`;
    // Ensure materials table and entry exist
    const label = mat.value?.label;
    if (!label) return;
    if (!sd.materials) sd.materials = {};
    if (!sd.materials[label]) sd.materials[label] = {};
    sd.materials[label][field] = texKey;
    // Trigger reactivity
    sceneData.value = { ...sd };
}

function removeSceneTexture(field: string) {
    const sd = sceneData.value;
    const label = mat.value?.label;
    if (!sd || !label || !sd.materials?.[label]) return;
    delete sd.materials[label][field];
    sceneData.value = { ...sd };
}

async function onPickTexture(file: string) {
    const field = pickerSlot.value;
    if (!field) return;
    pickerSlot.value = null;

    const handle = await props.engine.resources.loadImageToTexture(`/textures/${file}`);
    props.engine.materials.updateMaterial(props.materialHandle, {
        textures: { [field]: handle },
    });
    setSceneTexture(field, file);
    bump();
}

function onClearTexture() {
    const field = pickerSlot.value;
    if (!field) return;
    pickerSlot.value = null;

    props.engine.materials.updateMaterial(props.materialHandle, {
        textures: { [field]: undefined },
    });
    removeSceneTexture(field);
    bump();
}

function clearSlot(field: string) {
    props.engine.materials.updateMaterial(props.materialHandle, {
        textures: { [field]: undefined },
    });
    removeSceneTexture(field);
    bump();
}
</script>

<template>
    <div v-if="mat">
        <div class="prop-row">
            <span>Name</span>
            <span class="prop-value">{{ mat!.label }}</span>
        </div>
        <div class="prop-row">
            <span>Shading</span>
            <span class="prop-value">{{ mat!.shadingModel }}</span>
        </div>
        <div class="prop-row">
            <span>Alpha</span>
            <span class="prop-value">{{ mat!.alphaMode }}</span>
        </div>
        <div class="prop-row">
            <span>Render Path</span>
            <span class="prop-value">{{ mat!.renderPath }}</span>
        </div>

        <!-- Base Color -->
        <div class="prop-label" v-on="hint('baseColor')">Base Color (RGBA)</div>
        <div class="color-preview-row">
            <ColorSwatch
                :r="mat!.pbrParams.baseColorFactor[0]"
                :g="mat!.pbrParams.baseColorFactor[1]"
                :b="mat!.pbrParams.baseColorFactor[2]"
                @update:rgb="onSwatchUpdate"
            />
            <div class="color-inputs">
                <div class="vec3-row">
                    <NumericInput label="R" :model-value="mat!.pbrParams.baseColorFactor[0]" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => updateBaseColor(0, v)" />
                    <NumericInput label="G" :model-value="mat!.pbrParams.baseColorFactor[1]" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => updateBaseColor(1, v)" />
                </div>
                <div class="vec3-row">
                    <NumericInput label="B" :model-value="mat!.pbrParams.baseColorFactor[2]" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => updateBaseColor(2, v)" />
                    <NumericInput label="A" :model-value="mat!.pbrParams.baseColorFactor[3]" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => updateBaseColor(3, v)" />
                </div>
            </div>
        </div>

        <!-- Metallic / Roughness -->
        <div class="prop-label" v-on="hint('metallic')">Metallic</div>
        <div class="single-row">
            <NumericInput :model-value="mat!.pbrParams.metallicFactor" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => updatePBR('metallicFactor', v)" />
        </div>
        <div class="prop-label" v-on="hint('roughness')">Roughness</div>
        <div class="single-row">
            <NumericInput :model-value="mat!.pbrParams.roughnessFactor" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => updatePBR('roughnessFactor', v)" />
        </div>

        <!-- Emissive -->
        <div class="prop-label" v-on="hint('emissive')">Emissive (RGB)</div>
        <div class="vec3-row">
            <NumericInput label="R" :model-value="mat!.pbrParams.emissiveFactor[0]" :step="0.01" :min="0" :precision="3" @update:model-value="v => updateEmissive(0, v)" />
            <NumericInput label="G" :model-value="mat!.pbrParams.emissiveFactor[1]" :step="0.01" :min="0" :precision="3" @update:model-value="v => updateEmissive(1, v)" />
            <NumericInput label="B" :model-value="mat!.pbrParams.emissiveFactor[2]" :step="0.01" :min="0" :precision="3" @update:model-value="v => updateEmissive(2, v)" />
        </div>

        <!-- Normal Scale / Occlusion / Alpha Cutoff -->
        <div class="prop-label" v-on="hint('normalScale')">Normal Scale</div>
        <div class="single-row">
            <NumericInput :model-value="mat!.pbrParams.normalScale" :step="0.1" :min="0" :precision="2" @update:model-value="v => updatePBR('normalScale', v)" />
        </div>
        <div class="prop-label" v-on="hint('occlusionStrength')">Occlusion Strength</div>
        <div class="single-row">
            <NumericInput :model-value="mat!.pbrParams.occlusionStrength" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => updatePBR('occlusionStrength', v)" />
        </div>
        <div class="prop-label" v-if="mat!.alphaMode === 'MASK'" v-on="hint('alphaCutoff')">Alpha Cutoff</div>
        <div class="single-row" v-if="mat!.alphaMode === 'MASK'">
            <NumericInput :model-value="mat!.pbrParams.alphaCutoff" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => updatePBR('alphaCutoff', v)" />
        </div>

        <!-- Transparency (BLEND mode only) -->
        <template v-if="mat!.alphaMode === 'BLEND'">
            <div class="prop-label" v-on="hint('opacity')">Opacity</div>
            <div class="single-row">
                <NumericInput :model-value="mat!.pbrParams.opacity" :step="0.05" :min="0" :max="1" :precision="3" @update:model-value="v => updatePBR('opacity', v)" />
            </div>
            <div class="prop-label" v-on="hint('ior')">Index of Refraction</div>
            <div class="single-row">
                <NumericInput :model-value="mat!.pbrParams.ior" :step="0.05" :min="1" :max="3" :precision="2" @update:model-value="v => updatePBR('ior', v)" />
            </div>
            <div class="prop-row clickable" @click="toggleCastShadow" v-on="hint('castShadow')">
                <span>Cast Shadow</span>
                <span class="prop-value">
                    <i class="fas" :class="mat!.castShadow ? 'fa-toggle-on toggle-on' : 'fa-toggle-off toggle-off'"></i>
                </span>
            </div>
            <div class="prop-label" v-if="mat!.castShadow" v-on="hint('shadowOpacity')">Shadow Opacity</div>
            <div class="single-row" v-if="mat!.castShadow">
                <NumericInput :model-value="mat!.pbrParams.shadowOpacity" :step="0.05" :min="0" :max="1" :precision="2" @update:model-value="v => updatePBR('shadowOpacity', v)" />
            </div>
        </template>

        <!-- Toggles -->
        <div class="prop-row clickable" @click="toggleDoubleSided" v-on="hint('doubleSided')">
            <span>Double Sided</span>
            <span class="prop-value">
                <i class="fas" :class="mat!.doubleSided ? 'fa-toggle-on toggle-on' : 'fa-toggle-off toggle-off'"></i>
            </span>
        </div>

        <!-- Texture slots -->
        <div class="prop-label" v-on="hint('textures')"><i class="fas fa-image"></i> Textures</div>
        <div class="tex-list">
            <div
                v-for="s in texSlots"
                :key="s.field"
                class="tex-row"
                :class="{ assigned: hasTexture(s.field) }"
            >
                <div class="tex-preview" @click="openPicker(s.field)">
                    <img
                        v-if="texPreviewUrl(s.field)"
                        :src="texPreviewUrl(s.field)!"
                        :alt="s.label"
                        @error="($event.target as HTMLImageElement).style.display = 'none'"
                    />
                    <i v-else class="fas fa-image tex-placeholder-icon"></i>
                </div>
                <div class="tex-info" @click="openPicker(s.field)">
                    <span class="tex-slot-label">{{ s.label }}</span>
                    <span class="tex-file-name" v-if="hasTexture(s.field)">{{ texDisplayName(s.field) }}</span>
                    <span class="tex-file-name empty" v-else>None</span>
                </div>
                <button
                    v-if="hasTexture(s.field)"
                    class="tex-clear-btn"
                    title="Remove texture"
                    @click.stop="clearSlot(s.field)"
                >
                    <i class="fas fa-trash-alt"></i>
                </button>
            </div>
        </div>

        <TexturePicker
            v-if="pickerSlot"
            :slot="slotLabel(pickerSlot)"
            @pick="onPickTexture"
            @clear="onClearTexture"
            @close="pickerSlot = null"
        />
    </div>
</template>

<style scoped>
.prop-label { font-size: 11px; color: #888; margin-bottom: 4px; margin-top: 6px; display: flex; align-items: center; gap: 4px; cursor: help; }
.prop-row { display: flex; justify-content: space-between; padding: 2px 0; font-size: 12px; }
.prop-row span:first-child { color: #888; }
.prop-value { color: #ccc; }
.vec3-row { display: flex; gap: 6px; margin-bottom: 4px; }
.vec3-row > * { flex: 1; min-width: 0; }
.single-row { margin-bottom: 4px; }
.clickable { cursor: pointer; }
.clickable:hover { background: #333; border-radius: 3px; }
.toggle-on { color: #5b5; }
.toggle-off { color: #666; }

.color-preview-row { display: flex; gap: 8px; align-items: flex-start; }
.color-inputs { flex: 1; min-width: 0; }

.tex-list { display: flex; flex-direction: column; gap: 3px; }

.tex-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 2px 4px;
    border-radius: 4px;
    border: 1px solid transparent;
    transition: background 0.12s;
}
.tex-row:hover { background: #2a2a2a; }
.tex-row.assigned { border-color: #3a3a3a; }

.tex-preview {
    width: 28px;
    height: 28px;
    border-radius: 3px;
    background: #1a1a1a;
    border: 1px solid #333;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    cursor: pointer;
}
.tex-preview img { width: 100%; height: 100%; object-fit: cover; }
.tex-placeholder-icon { font-size: 10px; color: #444; }
.tex-preview:hover { border-color: #5588cc; }

.tex-info {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    cursor: pointer;
}
.tex-slot-label { font-size: 10px; color: #888; line-height: 1.2; }
.tex-file-name { font-size: 10px; color: #aaa; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; line-height: 1.2; }
.tex-file-name.empty { color: #555; font-style: italic; }

.tex-clear-btn {
    background: none;
    border: none;
    color: #664444;
    font-size: 10px;
    cursor: pointer;
    padding: 4px;
    border-radius: 3px;
    flex-shrink: 0;
}
.tex-clear-btn:hover { color: #cc6666; background: #3a2020; }
</style>
