<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue';

const props = defineProps<{
    r: number; // 0–1
    g: number;
    b: number;
}>();

const emit = defineEmits<{
    'update:rgb': [r: number, g: number, b: number];
    close: [];
}>();

// --- RGB <-> HSV conversion (all 0–1) ---
function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const d = max - min;
    const v = max;
    const s = max === 0 ? 0 : d / max;
    let h = 0;
    if (d !== 0) {
        if (max === r) h = ((g - b) / d + 6) % 6;
        else if (max === g) h = (b - r) / d + 2;
        else h = (r - g) / d + 4;
        h /= 6;
    }
    return [h, s, v];
}

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0: return [v, t, p];
        case 1: return [q, v, p];
        case 2: return [p, v, t];
        case 3: return [p, q, v];
        case 4: return [t, p, v];
        case 5: return [v, p, q];
    }
    return [v, t, p];
}

// Internal HSV state
const hue = ref(0);
const sat = ref(0);
const val = ref(0);

// Sync from props on mount / prop change
function syncFromProps() {
    const [h, s, v] = rgbToHsv(props.r, props.g, props.b);
    hue.value = h;
    sat.value = s;
    val.value = v;
}

watch(() => [props.r, props.g, props.b], syncFromProps);
onMounted(syncFromProps);

function emitColor() {
    const [r, g, b] = hsvToRgb(hue.value, sat.value, val.value);
    emit('update:rgb', r, g, b);
}

// Hue strip background (constant)
const hueGradient = 'linear-gradient(to right, #f00 0%, #ff0 17%, #0f0 33%, #0ff 50%, #00f 67%, #f0f 83%, #f00 100%)';

// SV area background: hue-based color
const svBackground = computed(() => {
    const [r, g, b] = hsvToRgb(hue.value, 1, 1);
    return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
});

// Hex display
const hexValue = computed(() => {
    const [r, g, b] = hsvToRgb(hue.value, sat.value, val.value);
    const toHex = (n: number) => Math.round(n * 255).toString(16).padStart(2, '0');
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
});

function onHexInput(e: Event) {
    const input = (e.target as HTMLInputElement).value.trim();
    const match = input.match(/^#?([0-9a-fA-F]{6})$/);
    if (!match) return;
    const hex = match[1];
    const r = parseInt(hex.slice(0, 2), 16) / 255;
    const g = parseInt(hex.slice(2, 4), 16) / 255;
    const b = parseInt(hex.slice(4, 6), 16) / 255;
    const [h, s, v] = rgbToHsv(r, g, b);
    hue.value = h;
    sat.value = s;
    val.value = v;
    emitColor();
}

// --- Dragging ---
const svAreaRef = ref<HTMLElement | null>(null);
const hueBarRef = ref<HTMLElement | null>(null);
const draggingSV = ref(false);
const draggingHue = ref(false);

function clamp01(v: number) { return Math.max(0, Math.min(1, v)); }

function onSVPointerDown(e: PointerEvent) {
    draggingSV.value = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    updateSV(e);
}

function onSVPointerMove(e: PointerEvent) {
    if (!draggingSV.value) return;
    updateSV(e);
}

function onSVPointerUp() {
    draggingSV.value = false;
}

function updateSV(e: PointerEvent) {
    const rect = svAreaRef.value!.getBoundingClientRect();
    sat.value = clamp01((e.clientX - rect.left) / rect.width);
    val.value = clamp01(1 - (e.clientY - rect.top) / rect.height);
    emitColor();
}

function onHuePointerDown(e: PointerEvent) {
    draggingHue.value = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    updateHue(e);
}

function onHuePointerMove(e: PointerEvent) {
    if (!draggingHue.value) return;
    updateHue(e);
}

function onHuePointerUp() {
    draggingHue.value = false;
}

function updateHue(e: PointerEvent) {
    const rect = hueBarRef.value!.getBoundingClientRect();
    hue.value = clamp01((e.clientX - rect.left) / rect.width);
    emitColor();
}

// Close on click outside
const rootRef = ref<HTMLElement | null>(null);

function onClickOutside(e: MouseEvent) {
    if (rootRef.value && !rootRef.value.contains(e.target as Node)) {
        emit('close');
    }
}

onMounted(() => {
    setTimeout(() => document.addEventListener('pointerdown', onClickOutside), 0);
});
onBeforeUnmount(() => {
    document.removeEventListener('pointerdown', onClickOutside);
});
</script>

<template>
    <div ref="rootRef" class="color-picker" @pointerdown.stop>
        <!-- SV area -->
        <div
            ref="svAreaRef"
            class="sv-area"
            :style="{ backgroundColor: svBackground }"
            @pointerdown="onSVPointerDown"
            @pointermove="onSVPointerMove"
            @pointerup="onSVPointerUp"
        >
            <div class="sv-white"></div>
            <div class="sv-black"></div>
            <div
                class="sv-cursor"
                :style="{ left: (sat * 100) + '%', top: ((1 - val) * 100) + '%' }"
            ></div>
        </div>

        <!-- Hue bar -->
        <div
            ref="hueBarRef"
            class="hue-bar"
            :style="{ background: hueGradient }"
            @pointerdown="onHuePointerDown"
            @pointermove="onHuePointerMove"
            @pointerup="onHuePointerUp"
        >
            <div
                class="hue-cursor"
                :style="{ left: (hue * 100) + '%' }"
            ></div>
        </div>

        <!-- Hex input -->
        <div class="hex-row">
            <span class="hex-label">Hex</span>
            <input
                class="hex-input"
                type="text"
                :value="hexValue"
                @change="onHexInput"
                spellcheck="false"
            />
        </div>
    </div>
</template>

<style scoped>
.color-picker {
    position: absolute;
    z-index: 100;
    background: #2a2a2a;
    border: 1px solid #555;
    border-radius: 6px;
    padding: 8px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    width: 200px;
    user-select: none;
}

.sv-area {
    width: 100%;
    height: 140px;
    border-radius: 4px;
    position: relative;
    cursor: crosshair;
    overflow: hidden;
}

.sv-white {
    position: absolute;
    inset: 0;
    background: linear-gradient(to right, #fff, transparent);
}

.sv-black {
    position: absolute;
    inset: 0;
    background: linear-gradient(to top, #000, transparent);
}

.sv-cursor {
    position: absolute;
    width: 12px;
    height: 12px;
    border: 2px solid #fff;
    border-radius: 50%;
    box-shadow: 0 0 2px rgba(0,0,0,0.6);
    transform: translate(-50%, -50%);
    pointer-events: none;
}

.hue-bar {
    width: 100%;
    height: 14px;
    border-radius: 3px;
    margin-top: 6px;
    position: relative;
    cursor: crosshair;
}

.hue-cursor {
    position: absolute;
    top: -1px;
    width: 6px;
    height: 16px;
    border: 2px solid #fff;
    border-radius: 3px;
    box-shadow: 0 0 2px rgba(0,0,0,0.6);
    transform: translateX(-50%);
    pointer-events: none;
}

.hex-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 6px;
}

.hex-label {
    font-size: 10px;
    color: #888;
}

.hex-input {
    flex: 1;
    background: #1e1e1e;
    border: 1px solid #444;
    border-radius: 3px;
    color: #ccc;
    font-size: 11px;
    font-family: monospace;
    padding: 2px 6px;
    outline: none;
    min-width: 0;
}
.hex-input:focus { border-color: #5588cc; }
</style>
