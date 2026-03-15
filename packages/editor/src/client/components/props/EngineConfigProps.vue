<script setup lang="ts">
import { ref, onMounted, inject, computed } from 'vue';
import type { Ref } from 'vue';
import { type Engine, type SSAOEffect, BackgroundType } from '@skengio/engine';
import NumericInput from '../NumericInput.vue';
import CollapsibleSection from '../CollapsibleSection.vue';
import ColorSwatch from '../ColorSwatch.vue';
import TexturePicker from '../TexturePicker.vue';
import { useHintPanel } from '../../composables/useHintPanel';
import { engineConfigHints } from '../../hints/engineConfigHints';

const props = defineProps<{
    engine: Engine;
}>();

const { hintEvents } = useHintPanel();
const hint = (key: string) => hintEvents(engineConfigHints[key]!);

// --- Reactive local copies (engine state is not Vue-reactive) ---
const frustumCulling = ref(true);
const frustumCullTolerance = ref(2.0);
const tonemapOperator = ref('ACES');

const ambientR = ref(0.03);
const ambientG = ref(0.03);
const ambientB = ref(0.03);

// SSAO
const ssaoEnabled = ref(true);
const ssaoRadius = ref(0.5);
const ssaoBias = ref(0.02);
const ssaoIntensity = ref(1.5);
const ssaoSamples = ref(16);
const ssaoBlurSharpness = ref(10.0);

// Background
const bgType = ref<BackgroundType>(BackgroundType.Gradient);
const bgColorR = ref(0);
const bgColorG = ref(0);
const bgColorB = ref(0);
const bgTopR = ref(0.1);
const bgTopG = ref(0.1);
const bgTopB = ref(0.3);
const bgBottomR = ref(0);
const bgBottomG = ref(0);
const bgBottomB = ref(0);

const tonemapOptions = ['ACES', 'AgX', 'Reinhard', 'PBR_Neutral'];
const bgTypeOptions = [BackgroundType.Color, BackgroundType.Gradient, BackgroundType.Texture, BackgroundType.Cubemap];

function readFromEngine() {
    const cfg = props.engine.config;
    frustumCulling.value = cfg.frustumCulling;
    frustumCullTolerance.value = cfg.frustumCullTolerance;
    tonemapOperator.value = cfg.tonemapOperator;

    // Ambient — read from orchestrator internal (best effort)
    const orch = props.engine.orchestrator as any;
    if (orch._ambientColor) {
        ambientR.value = orch._ambientColor[0];
        ambientG.value = orch._ambientColor[1];
        ambientB.value = orch._ambientColor[2];
    }

    // Background
    const bgCfg = props.engine.background.getConfig();
    bgType.value = bgCfg.type;
    if (bgCfg.color) {
        bgColorR.value = bgCfg.color[0];
        bgColorG.value = bgCfg.color[1];
        bgColorB.value = bgCfg.color[2];
    }
    if (bgCfg.topColor) {
        bgTopR.value = bgCfg.topColor[0];
        bgTopG.value = bgCfg.topColor[1];
        bgTopB.value = bgCfg.topColor[2];
    }
    if (bgCfg.bottomColor) {
        bgBottomR.value = bgCfg.bottomColor[0];
        bgBottomG.value = bgCfg.bottomColor[1];
        bgBottomB.value = bgCfg.bottomColor[2];
    }
    if (bgCfg.texturePath) {
        bgTextureName.value = bgCfg.texturePath;
    }
    if (bgCfg.cubemapBasePath) {
        // Strip leading /textures/ prefix if present, to match editor convention
        bgCubemapName.value = bgCfg.cubemapBasePath.replace(/^\/textures\//, '').replace(/\/$/, '');
        bgCubemapExt.value = bgCfg.cubemapExt ?? '.jpg';
    }

    // SSAO
    const ssao = props.engine.postProcess.getEffect<SSAOEffect>('SSAO');
    if (ssao) {
        ssaoEnabled.value = ssao.enabled;
        ssaoRadius.value = ssao.radius;
        ssaoBias.value = ssao.bias;
        ssaoIntensity.value = ssao.intensity;
        ssaoSamples.value = ssao.sampleCount;
        ssaoBlurSharpness.value = ssao.blurSharpness;
    }

}

// --- Updaters ---

function toggleFrustumCulling() {
    frustumCulling.value = !frustumCulling.value;
    props.engine.config.frustumCulling = frustumCulling.value;
}

function setFrustumTolerance(v: number) {
    frustumCullTolerance.value = v;
    props.engine.config.frustumCullTolerance = v;
}

function setTonemap(op: string) {
    tonemapOperator.value = op;
    props.engine.config.tonemapOperator = op;
}

function setAmbient(axis: number, v: number) {
    if (axis === 0) ambientR.value = v;
    else if (axis === 1) ambientG.value = v;
    else ambientB.value = v;
    props.engine.orchestrator.setAmbientColor(ambientR.value, ambientG.value, ambientB.value);
}

// Background updaters
function setBgType(t: BackgroundType) {
    bgType.value = t;
    applyBackground();
    // When switching away from cubemap, disable IBL reflections
    if (t !== BackgroundType.Cubemap) {
        props.engine.orchestrator.clearEnvironmentCubemap();
    }
}

function setBgColor(axis: number, v: number) {
    if (axis === 0) bgColorR.value = v;
    else if (axis === 1) bgColorG.value = v;
    else bgColorB.value = v;
    applyBackground();
}

function setBgTop(axis: number, v: number) {
    if (axis === 0) bgTopR.value = v;
    else if (axis === 1) bgTopG.value = v;
    else bgTopB.value = v;
    applyBackground();
}

function setBgBottom(axis: number, v: number) {
    if (axis === 0) bgBottomR.value = v;
    else if (axis === 1) bgBottomG.value = v;
    else bgBottomB.value = v;
    applyBackground();
}

function applyBackground() {
    const prev = props.engine.background.getConfig();
    props.engine.background.setConfig({
        type: bgType.value,
        color: [bgColorR.value, bgColorG.value, bgColorB.value],
        topColor: [bgTopR.value, bgTopG.value, bgTopB.value],
        bottomColor: [bgBottomR.value, bgBottomG.value, bgBottomB.value],
        textureHandle: prev.textureHandle,
        cubemapHandle: prev.cubemapHandle,
    });
}

// Background texture picker
const bgPickerOpen = ref(false);
const bgTextureName = ref<string | null>(null);
const bgCubemapName = ref<string | null>(null);

const textureFiles = inject<Ref<string[]>>('textureFiles', ref([]));

// ColorSwatch callbacks
function onBgColorSwatch(r: number, g: number, b: number) {
    bgColorR.value = r; bgColorG.value = g; bgColorB.value = b;
    applyBackground();
}
function onBgTopSwatch(r: number, g: number, b: number) {
    bgTopR.value = r; bgTopG.value = g; bgTopB.value = b;
    applyBackground();
}
function onBgBottomSwatch(r: number, g: number, b: number) {
    bgBottomR.value = r; bgBottomG.value = g; bgBottomB.value = b;
    applyBackground();
}

// Cubemap preview: use posz face
const bgCubemapExt = ref('.jpg');
const bgCubemapPreview = computed(() => {
    if (!bgCubemapName.value) return null;
    return `/textures/${bgCubemapName.value}/posz${bgCubemapExt.value}`;
});

const bgPickerFiles = computed(() => {
    if (bgType.value === BackgroundType.Cubemap) {
        return textureFiles.value.filter(f => f.startsWith('cubemaps/'));
    }
    return textureFiles.value.filter(f => !f.startsWith('cubemaps/'));
});

function openBgPicker() {
    bgPickerOpen.value = true;
}

async function onBgPickTexture(file: string) {
    bgPickerOpen.value = false;
    if (bgType.value === BackgroundType.Texture) {
        const handle = await props.engine.resources.loadImageToTexture(`/textures/${file}`);
        bgTextureName.value = file;
        const prev = props.engine.background.getConfig();
        props.engine.background.setConfig({ ...prev, type: BackgroundType.Texture, textureHandle: handle, texturePath: `/textures/${file}` });
    } else if (bgType.value === BackgroundType.Cubemap) {
        // Detect extension from first face
        const ext = await detectCubemapExt(file);
        bgCubemapExt.value = ext;
        const handle = await props.engine.resources.loadCubemapTexture(`/textures/${file}/`, ext);
        bgCubemapName.value = file;
        const prev = props.engine.background.getConfig();
        props.engine.background.setConfig({ ...prev, type: BackgroundType.Cubemap, cubemapHandle: handle, cubemapBasePath: `/textures/${file}/`, cubemapExt: ext });
        // Link the cubemap to the deferred lighting pass for IBL reflections
        props.engine.orchestrator.setEnvironmentCubemap(handle);
    }
}

async function detectCubemapExt(folder: string): Promise<string> {
    for (const ext of ['.jpg', '.png', '.webp']) {
        try {
            const res = await fetch(`/textures/${folder}/posx${ext}`, { method: 'HEAD' });
            if (res.ok) return ext;
        } catch { /* try next */ }
    }
    return '.jpg';
}

function onBgClearTexture() {
    bgPickerOpen.value = false;
    const prev = props.engine.background.getConfig();
    if (bgType.value === BackgroundType.Texture) {
        bgTextureName.value = null;
        props.engine.background.setConfig({ ...prev, textureHandle: undefined });
    } else if (bgType.value === BackgroundType.Cubemap) {
        bgCubemapName.value = null;
        props.engine.background.setConfig({ ...prev, cubemapHandle: undefined });
        props.engine.orchestrator.clearEnvironmentCubemap();
    }
}

// SSAO updaters
function toggleSSAO() {
    ssaoEnabled.value = !ssaoEnabled.value;
    const ssao = props.engine.postProcess.getEffect<SSAOEffect>('SSAO');
    if (ssao) ssao.enabled = ssaoEnabled.value;
}

function setSSAO(field: string, v: number) {
    const ssao = props.engine.postProcess.getEffect<SSAOEffect>('SSAO');
    if (!ssao) return;
    (ssao as any)[field] = v;
    switch (field) {
        case 'radius': ssaoRadius.value = v; break;
        case 'bias': ssaoBias.value = v; break;
        case 'intensity': ssaoIntensity.value = v; break;
        case 'sampleCount': ssaoSamples.value = v; break;
        case 'blurSharpness': ssaoBlurSharpness.value = v; break;
    }
}

onMounted(readFromEngine);
</script>

<template>
    <div class="engine-config">
        <!-- Rendering -->
        <CollapsibleSection title="Rendering" icon="fas fa-cogs" icon-color="#aabbcc">
            <div class="prop-row clickable" @click="toggleFrustumCulling" v-on="hint('frustumCulling')">
                <span>Frustum Culling</span>
                <span class="prop-value">
                    <i class="fas" :class="frustumCulling ? 'fa-toggle-on toggle-on' : 'fa-toggle-off toggle-off'"></i>
                </span>
            </div>
            <div class="prop-label" v-on="hint('frustumCullTolerance')">Cull Tolerance</div>
            <div class="single-row">
                <NumericInput :model-value="frustumCullTolerance" :step="0.5" :min="0" :precision="1" @update:model-value="setFrustumTolerance" />
            </div>
            <div class="prop-label" v-on="hint('tonemap')">Tonemap</div>
            <div class="tonemap-row">
                <button
                    v-for="op in tonemapOptions"
                    :key="op"
                    class="tonemap-btn"
                    :class="{ active: tonemapOperator === op }"
                    @click="setTonemap(op)"
                >{{ op }}</button>
            </div>
        </CollapsibleSection>

        <!-- Ambient -->
        <CollapsibleSection title="Ambient Light" icon="fas fa-sun" icon-color="#ffcc77">
            <div class="vec3-row" v-on="hint('ambient')">
                <NumericInput label="R" :model-value="ambientR" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => setAmbient(0, v)" />
                <NumericInput label="G" :model-value="ambientG" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => setAmbient(1, v)" />
                <NumericInput label="B" :model-value="ambientB" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => setAmbient(2, v)" />
            </div>
        </CollapsibleSection>

        <!-- Background -->
        <CollapsibleSection title="Background" icon="fas fa-image" icon-color="#88aacc">
            <div class="prop-label" v-on="hint('bgType')">Type</div>
            <div class="tonemap-row">
                <button
                    v-for="t in bgTypeOptions"
                    :key="t"
                    class="tonemap-btn"
                    :class="{ active: bgType === t }"
                    @click="setBgType(t)"
                >{{ t }}</button>
            </div>
            <!-- Color: single swatch + RGB sliders -->
            <template v-if="bgType === 'color'">
                <div class="prop-label" v-on="hint('bgColor')">Color</div>
                <div class="bg-preview-row">
                    <ColorSwatch :r="bgColorR" :g="bgColorG" :b="bgColorB" @update:rgb="onBgColorSwatch" />
                    <div class="bg-rgb-col">
                        <NumericInput label="R" :model-value="bgColorR" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => setBgColor(0, v)" />
                        <NumericInput label="G" :model-value="bgColorG" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => setBgColor(1, v)" />
                        <NumericInput label="B" :model-value="bgColorB" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => setBgColor(2, v)" />
                    </div>
                </div>
            </template>
            <!-- Gradient: two swatches with gradient preview + RGB sliders -->
            <template v-else-if="bgType === 'gradient'">
                <div class="prop-label" v-on="hint('bgGradientTop')">Top Color</div>
                <div class="bg-preview-row">
                    <ColorSwatch :r="bgTopR" :g="bgTopG" :b="bgTopB" @update:rgb="onBgTopSwatch" />
                    <div class="bg-rgb-col">
                        <NumericInput label="R" :model-value="bgTopR" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => setBgTop(0, v)" />
                        <NumericInput label="G" :model-value="bgTopG" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => setBgTop(1, v)" />
                        <NumericInput label="B" :model-value="bgTopB" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => setBgTop(2, v)" />
                    </div>
                </div>
                <div class="prop-label" v-on="hint('bgGradientBottom')">Bottom Color</div>
                <div class="bg-preview-row">
                    <ColorSwatch :r="bgBottomR" :g="bgBottomG" :b="bgBottomB" @update:rgb="onBgBottomSwatch" />
                    <div class="bg-rgb-col">
                        <NumericInput label="R" :model-value="bgBottomR" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => setBgBottom(0, v)" />
                        <NumericInput label="G" :model-value="bgBottomG" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => setBgBottom(1, v)" />
                        <NumericInput label="B" :model-value="bgBottomB" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => setBgBottom(2, v)" />
                    </div>
                </div>
            </template>
            <!-- Texture: image preview -->
            <template v-else-if="bgType === 'texture'">
                <div class="prop-label" v-on="hint('bgTexture')">Texture</div>
                <div class="bg-image-preview" @click="openBgPicker">
                    <img v-if="bgTextureName" :src="`/textures/${bgTextureName}`" :alt="bgTextureName" />
                    <span v-else class="bg-no-image">No texture selected</span>
                </div>
            </template>
            <!-- Cubemap: posz face preview -->
            <template v-else-if="bgType === 'cubemap'">
                <div class="prop-label" v-on="hint('bgCubemap')">Cubemap</div>
                <div class="bg-image-preview" @click="openBgPicker">
                    <img v-if="bgCubemapPreview" :src="bgCubemapPreview" :alt="bgCubemapName ?? ''" />
                    <span v-else class="bg-no-image">No cubemap selected</span>
                </div>
            </template>
            <TexturePicker
                v-if="bgPickerOpen"
                :slot="bgType === 'cubemap' ? 'Cubemap' : 'Texture'"
                :files="bgPickerFiles"
                @pick="onBgPickTexture"
                @clear="onBgClearTexture"
                @close="bgPickerOpen = false"
            />
        </CollapsibleSection>

        <!-- SSAO -->
        <CollapsibleSection title="SSAO" icon="fas fa-circle-half-stroke" icon-color="#99aacc">
            <div class="prop-row clickable" @click="toggleSSAO" v-on="hint('ssaoEnabled')">
                <span>Enabled</span>
                <span class="prop-value">
                    <i class="fas" :class="ssaoEnabled ? 'fa-toggle-on toggle-on' : 'fa-toggle-off toggle-off'"></i>
                </span>
            </div>
            <template v-if="ssaoEnabled">
                <div class="prop-label" v-on="hint('ssaoRadius')">Radius</div>
                <div class="single-row">
                    <NumericInput :model-value="ssaoRadius" :step="0.1" :min="0.1" :max="5" :precision="2" @update:model-value="v => setSSAO('radius', v)" />
                </div>
                <div class="prop-label" v-on="hint('ssaoBias')">Bias</div>
                <div class="single-row">
                    <NumericInput :model-value="ssaoBias" :step="0.005" :min="0" :max="0.2" :precision="3" @update:model-value="v => setSSAO('bias', v)" />
                </div>
                <div class="prop-label" v-on="hint('ssaoIntensity')">Intensity</div>
                <div class="single-row">
                    <NumericInput :model-value="ssaoIntensity" :step="0.1" :min="0" :max="5" :precision="2" @update:model-value="v => setSSAO('intensity', v)" />
                </div>
                <div class="prop-label" v-on="hint('ssaoSamples')">Samples</div>
                <div class="single-row">
                    <NumericInput :model-value="ssaoSamples" :step="4" :min="4" :max="32" :precision="0" @update:model-value="v => setSSAO('sampleCount', v)" />
                </div>
                <div class="prop-label" v-on="hint('ssaoBlurSharpness')">Blur Sharpness</div>
                <div class="single-row">
                    <NumericInput :model-value="ssaoBlurSharpness" :step="1" :min="0" :max="50" :precision="1" @update:model-value="v => setSSAO('blurSharpness', v)" />
                </div>
            </template>
        </CollapsibleSection>

        <!-- Read-only init-time config -->
        <CollapsibleSection title="Init-Time Config" icon="fas fa-lock" icon-color="#888" :collapsed="true">
            <div class="prop-row" v-on="hint('shadowAtlas')">
                <span>Shadow Atlas</span>
                <span class="prop-value">{{ engine.config.shadowAtlasSize }}px</span>
            </div>
            <div class="prop-row" v-on="hint('defaultShadowRes')">
                <span>Shadow Resolution</span>
                <span class="prop-value">{{ engine.config.defaultShadowMapResolution }}px</span>
            </div>
            <div class="prop-row" v-on="hint('defaultCascades')">
                <span>CSM Cascades</span>
                <span class="prop-value">{{ engine.config.defaultCsmCascades }}</span>
            </div>
            <div class="prop-row" v-on="hint('defaultBias')">
                <span>Shadow Bias</span>
                <span class="prop-value">{{ engine.config.defaultShadowBias }}</span>
            </div>
            <div class="prop-row" v-on="hint('maxLights')">
                <span>Max Lights</span>
                <span class="prop-value">{{ engine.config.maxLights }}</span>
            </div>
            <div class="prop-row" v-on="hint('lightsPerCluster')">
                <span>Lights/Cluster</span>
                <span class="prop-value">{{ engine.config.maxLightsPerCluster }}</span>
            </div>
            <div class="prop-row" v-on="hint('maxAnisotropy')">
                <span>Max Anisotropy</span>
                <span class="prop-value">{{ engine.config.maxAnisotropy }}x</span>
            </div>
            <div class="prop-row" v-on="hint('maxDraws')">
                <span>Max Draws/Frame</span>
                <span class="prop-value">{{ engine.config.maxDrawsPerFrame }}</span>
            </div>
            <div class="prop-row" v-on="hint('maxInstances')">
                <span>Max Instances</span>
                <span class="prop-value">{{ engine.config.maxStaticInstances }}</span>
            </div>
            <div class="prop-row" v-on="hint('halfResolution')">
                <span>Half Resolution</span>
                <span class="prop-value">
                    <i class="fas" :class="engine.config.halfResolution ? 'fa-toggle-on toggle-on' : 'fa-toggle-off toggle-off'"></i>
                </span>
            </div>
        </CollapsibleSection>
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

.tonemap-row { display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 4px; }
.tonemap-btn {
    background: #333;
    border: 1px solid #555;
    border-radius: 3px;
    color: #999;
    font-size: 10px;
    padding: 2px 6px;
    cursor: pointer;
}
.tonemap-btn:hover { background: #444; color: #ccc; }
.tonemap-btn.active { background: #2a4a7a; border-color: #5588cc; color: #ddd; }

.bg-preview-row { display: flex; gap: 8px; align-items: flex-start; margin-bottom: 4px; }
.bg-rgb-col { display: flex; flex-direction: column; gap: 2px; flex: 1; min-width: 0; }

.bg-image-preview {
    width: 100%;
    height: 64px;
    border: 1px solid #555;
    border-radius: 4px;
    overflow: hidden;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #1a1a1a;
    margin-bottom: 4px;
    transition: border-color 0.15s;
}
.bg-image-preview:hover { border-color: #88aadd; }
.bg-image-preview img { width: 100%; height: 100%; object-fit: cover; }
.bg-no-image { font-size: 10px; color: #555; }

.tex-grid { display: flex; flex-wrap: wrap; gap: 4px 12px; }
.tex-slot { font-size: 10px; color: #555; cursor: pointer; padding: 1px 4px; border-radius: 3px; }
.tex-slot:hover { background: #333; color: #aaa; }
.tex-slot i { font-size: 6px; margin-right: 3px; }
.tex-slot.active { color: #8c8; }
.tex-slot.active i { color: #5a5; }
.tex-slot.active:hover { color: #aea; }
</style>
