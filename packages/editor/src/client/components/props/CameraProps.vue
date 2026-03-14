<script setup lang="ts">
import { computed } from 'vue';
import { type Engine, ProjectionType } from '@skengio/engine';
import NumericInput from '../NumericInput.vue';
import { useEngineTick } from '../../composables/useEngineTick';
import { useHintPanel } from '../../composables/useHintPanel';
import { cameraHints } from '../../hints/cameraHints';

const props = defineProps<{
    engine: Engine;
}>();

const { tick, bump } = useEngineTick();
const { hintEvents } = useHintPanel();
const hint = (key: string) => hintEvents(cameraHints[key]!);

// ── Controller selector ──────────────────────────────────────────────────
const controllerNames = computed(() => {
    tick.value;
    return props.engine.cameraControllers.getAvailableNames();
});

const activeControllerName = computed(() => {
    tick.value;
    return props.engine.cameraControllers.getActive()?.name ?? null;
});

function setController(name: string) {
    props.engine.cameraControllers.setActive(name);
    bump();
}

// ── Device selector ─────────────────────────────────────────────────────
const activeDevice = computed(() => {
    tick.value;
    const name = activeControllerName.value;
    if (!name) return 'mouse';
    return props.engine.cameraControllers.getDevice(name) ?? 'mouse';
});

function setDevice(device: string) {
    const name = activeControllerName.value;
    if (!name) return;
    props.engine.cameraControllers.setDevice(name, device as 'mouse' | 'gamepad');
    bump();
}

// ── Axis inversion ──────────────────────────────────────────────────────
const isInvertX = computed(() => {
    tick.value;
    const name = activeControllerName.value;
    if (!name) return false;
    return props.engine.cameraControllers.getInvertX(name);
});

const isInvertY = computed(() => {
    tick.value;
    const name = activeControllerName.value;
    if (!name) return false;
    return props.engine.cameraControllers.getInvertY(name);
});

function toggleInvertX() {
    const name = activeControllerName.value;
    if (!name) return;
    props.engine.cameraControllers.setInvertX(name, !isInvertX.value);
    bump();
}

function toggleInvertY() {
    const name = activeControllerName.value;
    if (!name) return;
    props.engine.cameraControllers.setInvertY(name, !isInvertY.value);
    bump();
}

const cam = computed(() => {
    tick.value;
    const c = props.engine.cameras.getActiveCamera();
    return c ? { ...c } : undefined;
});

const isPerspective = computed(() => cam.value?.projectionType === ProjectionType.Perspective);

function updatePerspective(field: string, value: number) {
    const c = cam.value;
    if (!c) return;
    (c.perspective as any)[field] = value;
    props.engine.cameras.updateProjection(c.handle, c.perspective.aspectRatio);
    bump();
}

function updateOrtho(field: string, value: number) {
    const c = cam.value;
    if (!c) return;
    (c.orthographic as any)[field] = value;
    props.engine.cameras.updateProjection(c.handle, c.perspective.aspectRatio);
    bump();
}

function setExposure(value: number) {
    const c = cam.value;
    if (!c) return;
    props.engine.cameras.setExposure(c.handle, value);
    bump();
}

function toggleTAA() {
    const c = cam.value;
    if (!c) return;
    c.taaEnabled = !c.taaEnabled;
    bump();
}

const toDeg = 180 / Math.PI;
const toRad = Math.PI / 180;
</script>

<template>
    <div v-if="cam">
        <div class="prop-row" v-on="hint('projection')">
            <span>Projection</span>
            <span class="prop-value">{{ cam!.projectionType }}</span>
        </div>

        <!-- Perspective -->
        <template v-if="isPerspective">
            <div class="prop-label" v-on="hint('fovY')">FOV (degrees)</div>
            <div class="single-row">
                <NumericInput :model-value="cam!.perspective.fovY * toDeg" :step="1" :min="10" :max="170" :precision="1" @update:model-value="v => updatePerspective('fovY', v * toRad)" />
            </div>
            <div class="prop-label" v-on="hint('nearClip')">Near Clip</div>
            <div class="single-row">
                <NumericInput :model-value="cam!.perspective.near" :step="0.01" :min="0.001" :precision="3" @update:model-value="v => updatePerspective('near', v)" />
            </div>
            <div class="prop-label" v-on="hint('farClip')">Far Clip</div>
            <div class="single-row">
                <NumericInput :model-value="cam!.perspective.far" :step="10" :min="1" @update:model-value="v => updatePerspective('far', v)" />
            </div>
        </template>

        <!-- Orthographic -->
        <template v-else>
            <div class="prop-label" v-on="hint('orthoLR')">Left / Right</div>
            <div class="vec3-row">
                <NumericInput label="L" :model-value="cam!.orthographic.left" :step="1" @update:model-value="v => updateOrtho('left', v)" />
                <NumericInput label="R" :model-value="cam!.orthographic.right" :step="1" @update:model-value="v => updateOrtho('right', v)" />
            </div>
            <div class="prop-label" v-on="hint('orthoBT')">Bottom / Top</div>
            <div class="vec3-row">
                <NumericInput label="B" :model-value="cam!.orthographic.bottom" :step="1" @update:model-value="v => updateOrtho('bottom', v)" />
                <NumericInput label="T" :model-value="cam!.orthographic.top" :step="1" @update:model-value="v => updateOrtho('top', v)" />
            </div>
            <div class="prop-label" v-on="hint('orthoNF')">Near / Far</div>
            <div class="vec3-row">
                <NumericInput label="N" :model-value="cam!.orthographic.near" :step="0.1" :min="0.001" @update:model-value="v => updateOrtho('near', v)" />
                <NumericInput label="F" :model-value="cam!.orthographic.far" :step="10" :min="1" @update:model-value="v => updateOrtho('far', v)" />
            </div>
        </template>

        <!-- Exposure -->
        <div class="prop-label" v-on="hint('exposure')">Exposure (EV100)</div>
        <div class="single-row">
            <NumericInput :model-value="cam!.exposure" :step="0.1" @update:model-value="setExposure" />
        </div>

        <!-- TAA -->
        <div class="prop-row clickable" @click="toggleTAA" v-on="hint('taa')">
            <span>TAA</span>
            <span class="prop-value">
                <i class="fas" :class="cam!.taaEnabled ? 'fa-toggle-on toggle-on' : 'fa-toggle-off toggle-off'"></i>
            </span>
        </div>

        <!-- Controller -->
        <div v-if="controllerNames.length" class="prop-label" v-on="hint('controller')">Controller</div>
        <div v-if="controllerNames.length" class="controller-row">
            <select
                class="ctrl-select"
                :value="activeControllerName"
                @change="setController(($event.target as HTMLSelectElement).value)"
                v-on="hint('controller')"
            >
                <option v-for="name in controllerNames" :key="name" :value="name">{{ name }}</option>
            </select>
            <select
                class="ctrl-select device-select"
                :value="activeDevice"
                @change="setDevice(($event.target as HTMLSelectElement).value)"
                v-on="hint('device')"
            >
                <option value="mouse">Mouse</option>
                <option value="gamepad">Gamepad</option>
            </select>
        </div>
        <div v-if="controllerNames.length" class="invert-row">
            <div class="prop-row clickable" @click="toggleInvertX" v-on="hint('invertAxes')">
                <span>Invert X</span>
                <span class="prop-value">
                    <i class="fas" :class="isInvertX ? 'fa-toggle-on toggle-on' : 'fa-toggle-off toggle-off'"></i>
                </span>
            </div>
            <div class="prop-row clickable" @click="toggleInvertY" v-on="hint('invertAxes')">
                <span>Invert Y</span>
                <span class="prop-value">
                    <i class="fas" :class="isInvertY ? 'fa-toggle-on toggle-on' : 'fa-toggle-off toggle-off'"></i>
                </span>
            </div>
        </div>
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
.controller-row { display: flex; gap: 4px; margin-bottom: 4px; }
.ctrl-select {
    flex: 1; min-width: 0;
    padding: 3px 6px; font-size: 11px;
    background: #2a2a2a; color: #ccc; border: 1px solid #444;
    border-radius: 3px; cursor: pointer;
    outline: none;
}
.device-select { flex: 0 0 auto; width: 80px; }
.ctrl-select:hover { border-color: #5588cc; }
.ctrl-select:focus { border-color: #5588cc; background: #2a4a7a; color: #fff; }
.ctrl-select option { background: #2a2a2a; color: #ccc; }
.invert-row { display: flex; gap: 12px; margin-bottom: 4px; }
.invert-row .prop-row { flex: 1; }
</style>
