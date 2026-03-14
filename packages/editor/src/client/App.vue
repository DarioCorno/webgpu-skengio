<script setup lang="ts">
import { shallowRef, ref, provide, onMounted, onBeforeUnmount } from 'vue';
import { type Engine, type SSAOEffect, type SSREffect } from '@skengio/engine';
import Viewport from './components/Viewport.vue';
import Sidebar from './components/Sidebar.vue';
import MaterialPanel from './components/MaterialPanel.vue';
import StatsOverlay from './components/StatsOverlay.vue';
import MenuBar from './components/MenuBar.vue';
import AlertDialog from './components/AlertDialog.vue';
import HintPanel from './components/HintPanel.vue';
import { provideHintPanel } from './composables/useHintPanel';

const engine = shallowRef<Engine | null>(null);
const selectedNode = ref<number | null>(-1);
const viewportRef = ref<InstanceType<typeof Viewport> | null>(null);
const sceneName = ref('animations/gltf-boy');
// Key used to force remount of engine-dependent components after scene reload
const sceneKey = ref(0);
const textureFiles = ref<string[]>([]);

// Parsed scene JSON — persists across component re-mounts.
// MaterialProps reads/writes texture references here.
const sceneData = ref<Record<string, any>>({});

provide('engine', engine);
provide('selectedNode', selectedNode);
provide('textureFiles', textureFiles);
provide('sceneData', sceneData);

const isFullscreen = ref(false);

const hintState = provideHintPanel();

onMounted(async () => {
    try {
        const res = await fetch('/api/textures');
        textureFiles.value = await res.json();
    } catch { /* ignore */ }
});

function onEngineReady(e: Engine) {
    engine.value = e;
}

function onSelectNode(handle: number | null) {
    selectedNode.value = handle;
}

async function onOpenScene(file: string) {
    if (!viewportRef.value) return;
    selectedNode.value = -1;
    sceneName.value = file.replace(/\.json$/, '');
    await viewportRef.value.loadScene(`/scenes/${file}`);
}

function onSceneLoaded(data: Record<string, any>) {
    sceneData.value = data;
    // Bump key to force Sidebar, MaterialPanel, StatsOverlay to remount with fresh data
    sceneKey.value++;
}

function syncEngineStateToSceneData() {
    const e = engine.value;
    if (!e) return;

    const data = sceneData.value;

    // ── Camera ──────────────────────────────────────────────────────────
    const cam = e.cameras.getActiveCamera();
    if (cam) {
        if (!data.camera) data.camera = {};
        // Position & rotation from the camera's scene node
        if (cam.nodeHandle !== null) {
            const t = e.scene.getLocalTransform(cam.nodeHandle);
            if (t) {
                data.camera.position = [t.position[0], t.position[1], t.position[2]];
                data.camera.rotation = [t.rotation[0], t.rotation[1], t.rotation[2], t.rotation[3]];
            }
        }
        // Perspective params (fovY stored in radians → save as degrees)
        data.camera.fovY = cam.perspective.fovY * (180 / Math.PI);
        data.camera.near = cam.perspective.near;
        data.camera.far = cam.perspective.far;
        data.camera.exposure = cam.exposure;
        data.camera.taaEnabled = cam.taaEnabled;
    }

    // ── envConfig ───────────────────────────────────────────────────────
    if (!data.envConfig) data.envConfig = {};
    const env = data.envConfig;

    // Ambient
    const orch = e.orchestrator as any;
    if (orch._ambientColor) {
        env.ambient = [orch._ambientColor[0], orch._ambientColor[1], orch._ambientColor[2]];
    } else if (!env.ambient) {
        env.ambient = [0, 0, 0];
    }

    // SSAO
    const ssao = e.postProcess.getEffect<SSAOEffect>('SSAO');
    if (ssao) {
        env.ssao = {
            enabled: ssao.enabled,
            radius: ssao.radius,
            bias: ssao.bias,
            intensity: ssao.intensity,
            sampleCount: ssao.sampleCount,
            blurSharpness: ssao.blurSharpness,
        };
    } else if (!env.ssao) {
        env.ssao = { enabled: false };
    }

    // SSR
    const ssr = e.postProcess.getEffect<SSREffect>('SSR');
    if (ssr) {
        env.ssr = {
            enabled: ssr.enabled,
            maxRaySteps: ssr.maxRaySteps,
            thickness: ssr.thickness,
            stride: ssr.stride,
            fadeEnd: ssr.fadeEnd,
            roughnessCutoff: ssr.roughnessCutoff,
            jitterScale: ssr.jitterScale,
            maxDistance: ssr.maxDistance,
            strideZCutoff: ssr.strideZCutoff,
            envFallbackStr: ssr.envFallbackStr,
        };
    } else if (!env.ssr) {
        env.ssr = { enabled: false };
    }
}

async function onSaveScene() {
    syncEngineStateToSceneData();
    try {
        const res = await fetch(`/api/scenes/${sceneName.value}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(sceneData.value),
        });
        if (!res.ok) throw new Error(await res.text());
        saveResult.value = { success: true };
    } catch (e: any) {
        saveResult.value = { success: false, error: e.message };
    }
}

const saveResult = ref<{ success: boolean; error?: string } | null>(null);

function onSaveResultDismiss() {
    saveResult.value = null;
}

// ── Fullscreen ───────────────────────────────────────────────────────
function onFullscreenChange() {
    isFullscreen.value = !!document.fullscreenElement;
}

async function toggleFullscreen() {
    const el = document.documentElement;
    if (!document.fullscreenElement) {
        await el.requestFullscreen();
    } else {
        await document.exitFullscreen();
    }
}

onMounted(() => {
    document.addEventListener('fullscreenchange', onFullscreenChange);
});

onBeforeUnmount(() => {
    document.removeEventListener('fullscreenchange', onFullscreenChange);
});
</script>

<template>
    <MenuBar v-if="!isFullscreen" @open-scene="onOpenScene" @save-scene="onSaveScene" @fullscreen="toggleFullscreen" />
    <div class="main-row">
        <Viewport ref="viewportRef" :default-scene="sceneName" @engine-ready="onEngineReady" @scene-loaded="onSceneLoaded" />
        <Sidebar
            v-if="engine && !isFullscreen"
            :key="sceneKey"
            :engine="engine"
            :scene-name="sceneName"
            :selected-node="selectedNode"
            @select-node="onSelectNode"
        />
    </div>
    <HintPanel v-if="!isFullscreen" :state="hintState" />
    <MaterialPanel v-if="engine && !isFullscreen" :key="'mat-' + sceneKey" :engine="engine" />
    <StatsOverlay v-if="engine" :engine="engine" :fullscreen="isFullscreen" />
    <AlertDialog
        v-if="saveResult"
        :icon="saveResult.success ? 'fas fa-check-circle' : 'fas fa-exclamation-triangle'"
        :title="saveResult.success ? 'Scene Saved' : 'Save Failed'"
        :body="saveResult.success ? `Scene '${sceneName}' saved successfully.` : saveResult.error"
        confirm-text="OK"
        confirm-icon="fas fa-check"
        :show-cancel="false"
        @confirm="onSaveResultDismiss"
    />
</template>

<style scoped>
.main-row {
    display: flex;
    flex: 1;
    min-height: 0;
}
</style>
