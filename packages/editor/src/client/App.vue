<script setup lang="ts">
import { shallowRef, ref, provide, onMounted, onBeforeUnmount } from 'vue';
import {
    type Engine, type SSAOEffect, type SceneNode, type LightRecord,
    BackgroundType, NodeType, LightType, ShadowType, ProjectionType,
} from '@skengio/engine';
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
const sceneName = ref('grid');
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
    const RAD2DEG = 180 / Math.PI;

    // ── Engine config ───────────────────────────────────────────────────
    if (!data.engineConfig) data.engineConfig = {};
    data.engineConfig.frustumCulling = e.config.frustumCulling;
    data.engineConfig.tonemapOperator = e.config.tonemapOperator;

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
        // Projection type
        data.camera.projection = cam.projectionType === ProjectionType.Orthographic
            ? 'orthographic' : 'perspective';
        // Perspective params (fovY stored in radians → save as degrees)
        data.camera.fovY = cam.perspective.fovY * RAD2DEG;
        data.camera.near = cam.perspective.near;
        data.camera.far = cam.perspective.far;
        data.camera.exposure = cam.exposure;
        data.camera.taaEnabled = cam.taaEnabled;
        // Orthographic params (save alongside perspective so switching is non-destructive)
        if (cam.projectionType === ProjectionType.Orthographic) {
            data.camera.orthoLeft   = cam.orthographic.left;
            data.camera.orthoRight  = cam.orthographic.right;
            data.camera.orthoBottom = cam.orthographic.bottom;
            data.camera.orthoTop    = cam.orthographic.top;
        }
        // Controllers
        const ctrlNames = e.cameraControllers.getRegisteredNames();
        if (ctrlNames.length > 0) {
            const activeCtrl = e.cameraControllers.getActive();
            data.camera.controllers = ctrlNames.map(name => {
                const ctrl = e.cameraControllers.getController(name) as any;
                if (!ctrl) return { type: name.toLowerCase(), active: false };
                const json: Record<string, any> = {
                    type: ctrl.name?.toLowerCase() ?? name.toLowerCase(),
                    active: activeCtrl?.name === ctrl.name,
                    invertX: ctrl.invertX ?? false,
                    invertY: ctrl.invertY ?? false,
                };
                // FreeLook-specific
                if (ctrl.moveSpeed !== undefined)        json.moveSpeed = ctrl.moveSpeed;
                if (ctrl.lookSensitivity !== undefined)   json.lookSensitivity = ctrl.lookSensitivity;
                if (ctrl.sprintMultiplier !== undefined)  json.sprintMultiplier = ctrl.sprintMultiplier;
                // Orbit / Editor-specific
                if (ctrl.orbitSensitivity !== undefined)  json.orbitSensitivity = ctrl.orbitSensitivity;
                if (ctrl.panSensitivity !== undefined)    json.panSensitivity = ctrl.panSensitivity;
                if (ctrl.zoomSensitivity !== undefined)   json.zoomSensitivity = ctrl.zoomSensitivity;
                if (ctrl.distance !== undefined)          json.distance = ctrl.distance;
                if (ctrl.minDistance !== undefined)        json.minDistance = ctrl.minDistance;
                if (ctrl.maxDistance !== undefined)        json.maxDistance = ctrl.maxDistance;
                return json;
            });
        }
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

    // Background
    const bgCfg = e.background.getConfig();
    const bgJson: Record<string, any> = { type: bgCfg.type };
    switch (bgCfg.type) {
        case BackgroundType.Color:
            bgJson.color = bgCfg.color ?? [0, 0, 0];
            break;
        case BackgroundType.Gradient:
            bgJson.topColor    = bgCfg.topColor    ?? [0, 0, 0];
            bgJson.bottomColor = bgCfg.bottomColor ?? [0, 0, 0];
            break;
        case BackgroundType.Texture:
            if (bgCfg.texturePath && data.textures) {
                const texKey = Object.keys(data.textures).find(
                    k => data.textures[k] === bgCfg.texturePath,
                );
                if (texKey) bgJson.textureKey = texKey;
            }
            break;
        case BackgroundType.Cubemap:
            if (bgCfg.cubemapBasePath) bgJson.basePath = bgCfg.cubemapBasePath;
            if (bgCfg.cubemapExt)      bgJson.ext      = bgCfg.cubemapExt;
            break;
    }
    env.background = bgJson;

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


    // ── Node transforms + lights + animation ────────────────────────────
    // Build label → engine node lookup (first match wins for duplicate labels)
    const nodeByLabel = new Map<string, SceneNode>();
    for (const node of e.scene.getNodes()) {
        if (node.label && !nodeByLabel.has(node.label)) {
            nodeByLabel.set(node.label, node);
        }
    }

    if (Array.isArray(data.nodes)) {
        for (const nodeJson of data.nodes) {
            const name = nodeJson.name;
            if (!name) continue;
            const node = nodeByLabel.get(name);
            if (!node) continue;

            // Transform
            const t = e.scene.getLocalTransform(node.handle);
            if (t) {
                nodeJson.position = [t.position[0], t.position[1], t.position[2]];
                nodeJson.rotation = [t.rotation[0], t.rotation[1], t.rotation[2], t.rotation[3]];
                nodeJson.scale    = [t.scale[0], t.scale[1], t.scale[2]];
            }

            // Light
            if (node.lightHandle !== undefined && nodeJson.light) {
                const lr = e.lights.getLight(node.lightHandle) as LightRecord | undefined;
                if (lr) {
                    nodeJson.light.color     = [lr.color[0], lr.color[1], lr.color[2]];
                    nodeJson.light.intensity = lr.intensity;
                    nodeJson.light.range     = lr.range;
                    nodeJson.light.castShadow = lr.castShadow;
                    nodeJson.light.shadowBias = lr.shadowBias;
                    nodeJson.light.pcfRadius  = lr.pcfRadius;
                    if (lr.type === LightType.Spot) {
                        nodeJson.light.innerConeAngle = lr.innerConeAngle * RAD2DEG;
                        nodeJson.light.outerConeAngle = lr.outerConeAngle * RAD2DEG;
                    }
                    // Shadow type enum → JSON string
                    if (lr.castShadow) {
                        nodeJson.light.shadowType =
                            lr.shadowType === ShadowType.Cascaded ? 'cascaded'
                          : lr.shadowType === ShadowType.Cube     ? 'cube'
                          : 'standard';
                        nodeJson.light.numCascades         = lr.numCascades;
                        nodeJson.light.shadowMapResolution = lr.shadowMapResolution;
                    }
                }
            }

            // Animation
            const pb = e.animations.getPlaybackState(node.handle);
            if (pb) {
                if (!nodeJson.animation) nodeJson.animation = {};
                nodeJson.animation.clip  = pb.clipName || pb.clipHandle;
                nodeJson.animation.speed = pb.speed;
                nodeJson.animation.loop  = pb.loop;
                nodeJson.animation.enabled = true;
            }
        }
    }

    // ── Materials (PBR params + flags) ──────────────────────────────────
    if (!data.materials) data.materials = {};
    for (const mr of e.materials.getMaterials()) {
        const label = mr.label;
        if (!label) continue;
        if (!data.materials[label]) data.materials[label] = {};
        const mj = data.materials[label];

        // PBR scalar params
        const p = mr.pbrParams;
        mj.baseColorFactor      = [...p.baseColorFactor];
        mj.metallicFactor       = p.metallicFactor;
        mj.roughnessFactor      = p.roughnessFactor;
        mj.emissiveFactor       = [...p.emissiveFactor];
        mj.normalScale          = p.normalScale;
        mj.occlusionStrength    = p.occlusionStrength;
        mj.alphaCutoff          = p.alphaCutoff;
        mj.opacity              = p.opacity;
        mj.ior                  = p.ior;
        mj.shadowOpacity        = p.shadowOpacity;

        // Flags
        mj.alphaMode    = mr.alphaMode.toLowerCase();
        mj.doubleSided  = mr.doubleSided;
        mj.castShadow   = mr.castShadow;

        // Texture keys are maintained by MaterialProps via sceneData direct
        // mutation — don't overwrite them here (we don't have the reverse
        // ResourceHandle → textureKey mapping).
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
