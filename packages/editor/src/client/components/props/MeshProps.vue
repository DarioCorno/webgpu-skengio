<script setup lang="ts">
import type { Engine } from '@skengio/engine';

const props = defineProps<{
    engine: Engine;
    meshHandle: number;
    materialHandles: number[];
}>();

function meshData() {
    return props.engine.meshes.getDrawData(props.meshHandle);
}

function aabb() {
    return props.engine.meshes.getAABB(props.meshHandle);
}

function boundingSphere() {
    return props.engine.meshes.getBoundingSphere(props.meshHandle);
}

function materialLabel(h: number): string {
    return props.engine.materials.getMaterial(h)?.label ?? `#${h}`;
}

function totalTriangles(): number {
    const dd = meshData();
    if (!dd) return 0;
    const lod0 = dd.lodLevels[0];
    if (!lod0) return 0;
    let total = 0;
    for (const sm of lod0.subMeshes) {
        total += (sm.indexCount > 0 ? sm.indexCount : sm.vertexCount) / 3;
    }
    return Math.floor(total);
}

function formatNum(n: number): string {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
    return String(n);
}
</script>

<template>
    <div v-if="meshData()">
        <div class="prop-row">
            <span>Triangles</span>
            <span class="prop-value">{{ formatNum(totalTriangles()) }}</span>
        </div>

        <div class="prop-row">
            <span>LOD Levels</span>
            <span class="prop-value">{{ meshData()!.lodLevels.length }}</span>
        </div>

        <div class="prop-row">
            <span>Sub-meshes</span>
            <span class="prop-value">{{ meshData()!.lodLevels[0]?.subMeshes.length ?? 0 }}</span>
        </div>

        <div class="prop-row" v-if="aabb()">
            <span>AABB min</span>
            <span class="prop-value small">{{ aabb()!.min.x.toFixed(2) }}, {{ aabb()!.min.y.toFixed(2) }}, {{ aabb()!.min.z.toFixed(2) }}</span>
        </div>
        <div class="prop-row" v-if="aabb()">
            <span>AABB max</span>
            <span class="prop-value small">{{ aabb()!.max.x.toFixed(2) }}, {{ aabb()!.max.y.toFixed(2) }}, {{ aabb()!.max.z.toFixed(2) }}</span>
        </div>

        <div class="prop-row" v-if="boundingSphere()">
            <span>Bounding R</span>
            <span class="prop-value">{{ boundingSphere()!.radius.toFixed(2) }}</span>
        </div>

        <!-- Materials assigned -->
        <div class="prop-label" v-if="materialHandles.length">
            <i class="fas fa-palette"></i> Materials ({{ materialHandles.length }})
        </div>
        <div class="mat-list" v-if="materialHandles.length">
            <div class="mat-item" v-for="(mh, i) in materialHandles" :key="mh">
                <span class="mat-index">[{{ i }}]</span> {{ materialLabel(mh) }}
            </div>
        </div>
    </div>
</template>

<style scoped>
.prop-label { font-size: 11px; color: #888; margin-bottom: 4px; margin-top: 6px; display: flex; align-items: center; gap: 4px; }
.prop-row { display: flex; justify-content: space-between; padding: 2px 0; font-size: 12px; }
.prop-row span:first-child { color: #888; }
.prop-value { color: #ccc; }
.prop-value.small { font-size: 10px; font-family: 'Consolas', monospace; }
.mat-list { margin-top: 2px; }
.mat-item { font-size: 11px; color: #aaa; padding: 1px 0; }
.mat-index { color: #666; font-family: 'Consolas', monospace; font-size: 10px; }
</style>
