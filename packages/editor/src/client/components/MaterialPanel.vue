<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue';
import type { Engine, MaterialRecord } from '@skengio/engine';

const props = defineProps<{
    engine: Engine;
}>();

const materials = ref<MaterialRecord[]>([]);
let refreshTimer = 0;

function refresh() {
    materials.value = Array.from(props.engine.materials.getMaterials());
}

onMounted(() => {
    refresh();
    refreshTimer = window.setInterval(refresh, 2000);
});

onBeforeUnmount(() => {
    clearInterval(refreshTimer);
});

function colorToCSS(c: [number, number, number, number]): string {
    const r = Math.round(c[0] * 255);
    const g = Math.round(c[1] * 255);
    const b = Math.round(c[2] * 255);
    return `rgb(${r},${g},${b})`;
}

function typeIcon(mat: MaterialRecord): string {
    if (mat.shadingModel === 'UNLIT') return 'fa-sun';
    if (mat.alphaMode === 'BLEND') return 'fa-glass-water';
    if (mat.pbrParams.metallicFactor > 0.5) return 'fa-gem';
    return 'fa-circle';
}
</script>

<template>
    <div class="material-panel">
        <div class="panel-header">
            <i class="fas fa-palette"></i> Materials
            <span class="mat-count">{{ materials.length }}</span>
        </div>
        <div class="mat-grid">
            <div
                v-for="mat in materials"
                :key="mat.handle"
                class="mat-card"
                :title="`${mat.label}\n${mat.shadingModel} · ${mat.alphaMode}\nMetallic: ${mat.pbrParams.metallicFactor.toFixed(2)}\nRoughness: ${mat.pbrParams.roughnessFactor.toFixed(2)}`"
            >
                <div
                    class="mat-preview"
                    :style="{ background: colorToCSS(mat.pbrParams.baseColorFactor) }"
                >
                    <i class="fas" :class="typeIcon(mat)"></i>
                </div>
                <div class="mat-name">{{ mat.label }}</div>
            </div>
        </div>
    </div>
</template>

<style scoped>
.material-panel {
    background: #252525;
    border-top: 1px solid #3a3a3a;
    display: flex;
    flex-direction: column;
    min-height: 0;
}

.panel-header {
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #999;
    background: #2d2d2d;
    border-bottom: 1px solid #3a3a3a;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
}

.mat-count {
    background: #3a3a3a;
    color: #888;
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 8px;
    margin-left: 4px;
}

.mat-grid {
    display: flex;
    gap: 8px;
    padding: 8px 12px;
    overflow-x: auto;
    overflow-y: hidden;
    flex: 1;
    align-items: flex-start;
}

.mat-card {
    width: 72px;
    min-width: 72px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    cursor: pointer;
}

.mat-card:hover .mat-preview {
    border-color: #5588cc;
}

.mat-preview {
    width: 64px;
    height: 64px;
    border-radius: 6px;
    border: 2px solid #444;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: border-color 0.15s;
}

.mat-preview i {
    font-size: 20px;
    color: rgba(255, 255, 255, 0.5);
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
}

.mat-name {
    font-size: 10px;
    color: #aaa;
    text-align: center;
    width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
</style>
