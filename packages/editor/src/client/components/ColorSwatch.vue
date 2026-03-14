<script setup lang="ts">
import { ref } from 'vue';
import ColorPicker from './ColorPicker.vue';

defineProps<{
    r: number;
    g: number;
    b: number;
}>();

const emit = defineEmits<{
    'update:rgb': [r: number, g: number, b: number];
}>();

const open = ref(false);

function onUpdate(r: number, g: number, b: number) {
    emit('update:rgb', r, g, b);
}
</script>

<template>
    <div class="swatch-wrapper">
        <div
            class="color-swatch"
            :style="{ background: `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})` }"
            title="Click to open color picker"
            @click="open = !open"
        ></div>
        <ColorPicker
            v-if="open"
            :r="r"
            :g="g"
            :b="b"
            @update:rgb="onUpdate"
            @close="open = false"
        />
    </div>
</template>

<style scoped>
.swatch-wrapper { position: relative; flex-shrink: 0; }
.color-swatch {
    width: 32px;
    height: 32px;
    border-radius: 4px;
    border: 1px solid #555;
    margin-top: 2px;
    cursor: pointer;
    transition: border-color 0.15s;
}
.color-swatch:hover { border-color: #88aadd; }
</style>
