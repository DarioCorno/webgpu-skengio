<script setup lang="ts">
import { computed } from 'vue';
import type { Engine } from '@skengio/engine';
import NumericInput from '../NumericInput.vue';
import { useEngineTick } from '../../composables/useEngineTick';

const props = defineProps<{
    engine: Engine;
    nodeHandle: number;
}>();

const { tick, bump } = useEngineTick();

const transform = computed(() => {
    tick.value; // reactive dependency
    const t = props.engine.scene.getLocalTransform(props.nodeHandle);
    // Shallow-clone so Vue detects a new reference after bump()
    return t ? { ...t } : undefined;
});

function updatePosition(axis: number, value: number) {
    const t = transform.value;
    if (!t) return;
    const pos = new Float32Array(t.position);
    pos[axis] = value;
    props.engine.scene.setLocalTransform(props.nodeHandle, { position: pos });
    bump();
}

function updateRotation(axis: number, degrees: number) {
    const t = transform.value;
    if (!t) return;
    const q = t.rotation;
    const euler = quatToEulerDeg(q);
    euler[axis] = degrees;
    const newQ = eulerDegToQuat(euler);
    props.engine.scene.setLocalTransform(props.nodeHandle, { rotation: newQ });
    bump();
}

function updateScale(axis: number, value: number) {
    const t = transform.value;
    if (!t) return;
    const scl = new Float32Array(t.scale);
    scl[axis] = value;
    props.engine.scene.setLocalTransform(props.nodeHandle, { scale: scl });
    bump();
}

// Quaternion ↔ Euler helpers (degrees, XYZ order)
function quatToEulerDeg(q: Float32Array): [number, number, number] {
    const [x, y, z, w] = q;
    const sinr = 2 * (w * x + y * z);
    const cosr = 1 - 2 * (x * x + y * y);
    const roll = Math.atan2(sinr, cosr);

    const sinp = 2 * (w * y - z * x);
    const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : Math.asin(sinp);

    const siny = 2 * (w * z + x * y);
    const cosy = 1 - 2 * (y * y + z * z);
    const yaw = Math.atan2(siny, cosy);

    const toDeg = 180 / Math.PI;
    return [roll * toDeg, pitch * toDeg, yaw * toDeg];
}

function eulerDegToQuat(euler: [number, number, number]): Float32Array {
    const toRad = Math.PI / 180;
    const [rx, ry, rz] = euler.map(d => d * toRad);
    const cx = Math.cos(rx * 0.5), sx = Math.sin(rx * 0.5);
    const cy = Math.cos(ry * 0.5), sy = Math.sin(ry * 0.5);
    const cz = Math.cos(rz * 0.5), sz = Math.sin(rz * 0.5);
    return new Float32Array([
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
        cx * cy * cz + sx * sy * sz,
    ]);
}
</script>

<template>
    <div v-if="transform">
        <div class="prop-label">Position</div>
        <div class="vec3-row">
            <NumericInput label="X" :model-value="transform!.position[0]" :step="0.1" @update:model-value="v => updatePosition(0, v)" />
            <NumericInput label="Y" :model-value="transform!.position[1]" :step="0.1" @update:model-value="v => updatePosition(1, v)" />
            <NumericInput label="Z" :model-value="transform!.position[2]" :step="0.1" @update:model-value="v => updatePosition(2, v)" />
        </div>

        <div class="prop-label">Rotation</div>
        <div class="vec3-row">
            <NumericInput label="X" :model-value="quatToEulerDeg(transform!.rotation)[0]" :step="1" @update:model-value="v => updateRotation(0, v)" />
            <NumericInput label="Y" :model-value="quatToEulerDeg(transform!.rotation)[1]" :step="1" @update:model-value="v => updateRotation(1, v)" />
            <NumericInput label="Z" :model-value="quatToEulerDeg(transform!.rotation)[2]" :step="1" @update:model-value="v => updateRotation(2, v)" />
        </div>

        <div class="prop-label">Scale</div>
        <div class="vec3-row">
            <NumericInput label="X" :model-value="transform!.scale[0]" :step="0.1" :min="0.01" @update:model-value="v => updateScale(0, v)" />
            <NumericInput label="Y" :model-value="transform!.scale[1]" :step="0.1" :min="0.01" @update:model-value="v => updateScale(1, v)" />
            <NumericInput label="Z" :model-value="transform!.scale[2]" :step="0.1" :min="0.01" @update:model-value="v => updateScale(2, v)" />
        </div>
    </div>
</template>

<style scoped>
.prop-label { font-size: 11px; color: #888; margin-bottom: 4px; display: flex; align-items: center; gap: 4px; }
.vec3-row { display: flex; gap: 6px; margin-bottom: 8px; }
.vec3-row > * { flex: 1; min-width: 0; }
</style>
