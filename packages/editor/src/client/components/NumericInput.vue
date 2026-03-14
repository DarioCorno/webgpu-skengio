<script setup lang="ts">
import { ref, computed, watch, onBeforeUnmount } from 'vue';

const props = withDefaults(defineProps<{
    modelValue: number;
    label?: string;
    min?: number;
    max?: number;
    step?: number;
    /** Multiplier applied to step when dragging (pixels per step). Default 1. */
    dragScale?: number;
    /** Decimal places shown when not focused. */
    precision?: number;
}>(), {
    step: 0.1,
    dragScale: 1,
    precision: 2,
});

const emit = defineEmits<{
    'update:modelValue': [value: number];
}>();

const inputRef = ref<HTMLInputElement | null>(null);
const editing = ref(false);
const editText = ref('');
const dragging = ref(false);

/**
 * Local value that tracks what this input should display.
 * During editing/dragging it is updated directly for immediate feedback.
 * When idle, it syncs from props.modelValue.
 */
const localValue = ref(props.modelValue);

const isActive = computed(() => editing.value || dragging.value);

const displayValue = computed(() => localValue.value.toFixed(props.precision));

// Sync from parent ONLY when idle — avoids overwriting mid-edit values
watch(() => props.modelValue, (v) => {
    if (!isActive.value) {
        localValue.value = v;
    }
});

// --- Keyboard editing ---

function onFocus() {
    editing.value = true;
    editText.value = localValue.value.toString();
}

function onInput(e: Event) {
    editText.value = (e.target as HTMLInputElement).value;
}

function commitEdit() {
    editing.value = false;
    const parsed = parseFloat(editText.value);
    if (!isNaN(parsed)) {
        const clamped = clamp(parsed);
        localValue.value = clamped;
        emit('update:modelValue', clamped);
    }
    // After edit ends, re-sync from prop on next tick
    requestAnimationFrame(() => {
        if (!isActive.value) localValue.value = props.modelValue;
    });
}

function onKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
        commitEdit();
        inputRef.value?.blur();
    } else if (e.key === 'Escape') {
        editing.value = false;
        localValue.value = props.modelValue;
        editText.value = props.modelValue.toString();
        inputRef.value?.blur();
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        nudge(1);
    } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        nudge(-1);
    }
}

function nudge(direction: number) {
    const next = clamp(localValue.value + direction * props.step);
    localValue.value = next;
    emit('update:modelValue', next);
    if (editing.value) {
        editText.value = next.toString();
    }
}

// --- Drag on label ---

let dragStartX = 0;
let dragStartValue = 0;

function onLabelPointerDown(e: PointerEvent) {
    if (e.button !== 0) return;
    dragging.value = true;
    dragStartX = e.clientX;
    dragStartValue = localValue.value;

    const target = e.currentTarget as HTMLElement;
    target.setPointerCapture(e.pointerId);
    target.addEventListener('pointermove', onLabelPointerMove);
    target.addEventListener('pointerup', onLabelPointerUp);
}

function onLabelPointerMove(e: PointerEvent) {
    if (!dragging.value) return;
    const dx = e.clientX - dragStartX;
    const steps = Math.round(dx / Math.max(props.dragScale, 0.1));
    const next = clamp(dragStartValue + steps * props.step);
    localValue.value = next;
    emit('update:modelValue', next);
}

function onLabelPointerUp(e: PointerEvent) {
    dragging.value = false;
    const target = e.currentTarget as HTMLElement;
    target.releasePointerCapture(e.pointerId);
    target.removeEventListener('pointermove', onLabelPointerMove);
    target.removeEventListener('pointerup', onLabelPointerUp);
    // Re-sync from engine after drag ends
    requestAnimationFrame(() => {
        if (!isActive.value) localValue.value = props.modelValue;
    });
}

// --- Helpers ---

function clamp(v: number): number {
    if (props.min !== undefined && v < props.min) v = props.min;
    if (props.max !== undefined && v > props.max) v = props.max;
    // Round to step precision to avoid floating point drift
    const inv = 1 / props.step;
    return Math.round(v * inv) / inv;
}

onBeforeUnmount(() => {
    dragging.value = false;
});
</script>

<template>
    <div class="numeric-input" :class="{ dragging }">
        <span
            v-if="label"
            class="ni-label"
            :class="{ dragging }"
            @pointerdown="onLabelPointerDown"
        >{{ label }}</span>
        <input
            ref="inputRef"
            type="number"
            class="ni-field"
            :value="editing ? editText : displayValue"
            :min="min"
            :max="max"
            :step="step"
            @focus="onFocus"
            @blur="commitEdit"
            @input="onInput"
            @keydown="onKeydown"
        />
    </div>
</template>

<style scoped>
.numeric-input {
    display: flex;
    align-items: center;
    gap: 4px;
    flex: 1;
    min-width: 0;
}

.ni-label {
    font-size: 11px;
    color: #888;
    cursor: ew-resize;
    user-select: none;
    flex-shrink: 0;
    padding: 2px 2px;
    border-radius: 2px;
}

.ni-label:hover {
    background: #3a3a3a;
    color: #bbb;
}

.ni-label.dragging {
    background: #445566;
    color: #ddd;
}

.ni-field {
    width: 100%;
    min-width: 0;
    background: #1a1a1a;
    border: 1px solid #444;
    border-radius: 3px;
    color: #ddd;
    padding: 3px 5px;
    font-size: 12px;
    font-family: 'Consolas', monospace;
    text-align: right;
}

.ni-field:focus {
    border-color: #5588cc;
    outline: none;
    text-align: left;
}

</style>
