<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue';
import type { Engine } from '@skengio/engine';

const props = defineProps<{
    engine: Engine;
    fullscreen?: boolean;
}>();

const visible = ref(!props.fullscreen);

watch(() => props.fullscreen, (fs) => { if (fs) visible.value = false; });

// Stats values
const realFps = ref(0);
const engineFps = ref(0);
const cpuTimeMs = ref(0);
const gpuTimeMs = ref(0);
const drawCalls = ref(0);
const triangles = ref(0);
const visibleObjects = ref(0);
const totalMeshObjects = ref(0);
const renderPasses = ref(0);
const gpuBuffers = ref(0);
const gpuTextures = ref(0);
const frameIndex = ref(0);

// EMA smoothing
let smoothCpu = 0;
let smoothGpu = 0;
const ALPHA = 0.1;

let rafId = 0;

function sample() {
    const eng = props.engine.orchestrator.getStats();
    const cull = props.engine.orchestrator.getCullResults();
    const res = props.engine.resources.getStats();

    smoothCpu = smoothCpu === 0 ? eng.cpuTimeMs : smoothCpu * (1 - ALPHA) + eng.cpuTimeMs * ALPHA;
    smoothGpu = smoothGpu === 0 ? eng.gpuTimeMs : smoothGpu * (1 - ALPHA) + eng.gpuTimeMs * ALPHA;

    const bottleneck = Math.max(smoothCpu, smoothGpu, 0.01);

    realFps.value = Math.round(eng.realFps * 10) / 10;
    engineFps.value = Math.round((1000 / bottleneck) * 10) / 10;
    cpuTimeMs.value = Math.round(smoothCpu * 10) / 10;
    gpuTimeMs.value = Math.round(smoothGpu * 10) / 10;
    drawCalls.value = eng.drawCalls;
    triangles.value = eng.triangles;
    visibleObjects.value = cull.opaqueDrawables.length + cull.transparentDrawables.length;
    totalMeshObjects.value = props.engine.scene.getMeshNodeCount();
    renderPasses.value = eng.passCount;
    gpuBuffers.value = res.buffers;
    gpuTextures.value = res.textures + res.pooledTextures;
    frameIndex.value = eng.frameIndex;

    rafId = requestAnimationFrame(sample);
}

function formatTris(n: number): string {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
    return String(n);
}

function rj(s: string, width: number): string {
    return s.length >= width ? s : ' '.repeat(width - s.length) + s;
}

const overlayText = computed(() => [
    `FPS      ${rj(realFps.value.toFixed(1), 7)}`,
    `Max FPS  ${rj(engineFps.value.toFixed(1), 7)}`,
    `CPU      ${rj(cpuTimeMs.value.toFixed(1), 6)} ms`,
    `GPU      ${rj(gpuTimeMs.value.toFixed(1), 6)} ms`,
    `─────────────────`,
    `Draw     ${rj(String(drawCalls.value), 7)}`,
    `Tris     ${rj(formatTris(triangles.value), 7)}`,
    `Objects  ${visibleObjects.value} / ${totalMeshObjects.value}`,
    `Passes   ${rj(String(renderPasses.value), 7)}`,
    `─────────────────`,
    `Buffers  ${rj(String(gpuBuffers.value), 7)}`,
    `Textures ${rj(String(gpuTextures.value), 7)}`,
    `Frame #  ${frameIndex.value}`,
    `─────────────────`,
    `[I] toggle`,
].join('\n'));

function onKeyDown(e: KeyboardEvent) {
    if (e.code === 'KeyI' && !e.repeat) {
        visible.value = !visible.value;
    }
}

onMounted(() => {
    rafId = requestAnimationFrame(sample);
    window.addEventListener('keydown', onKeyDown);
});

onBeforeUnmount(() => {
    cancelAnimationFrame(rafId);
    window.removeEventListener('keydown', onKeyDown);
});
</script>

<template>
    <div v-show="visible" class="stats-overlay">{{ overlayText }}</div>
</template>

<style scoped>
.stats-overlay {
    position: fixed;
    top: 30px;
    left: 10px;
    font-family: monospace;
    font-size: 11px;
    line-height: 1.6;
    color: #c8ffc8;
    background: rgba(0, 0, 0, 0.55);
    padding: 6px 10px;
    border-radius: 4px;
    pointer-events: none;
    user-select: none;
    white-space: pre;
    z-index: 50;
}
</style>
