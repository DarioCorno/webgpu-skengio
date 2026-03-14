<script setup lang="ts">
import type { HintPanelState } from '../composables/useHintPanel';

defineProps<{
    state: HintPanelState;
}>();
</script>

<template>
    <Transition name="hint-fade">
        <div
            v-if="state.visible.value && state.content.value"
            class="hint-panel"
            :style="state.content.value.anchorY != null
                ? { top: state.content.value.anchorY + 'px' }
                : {}"
        >
            <div class="hint-title">
                <i class="fas fa-info-circle"></i>
                {{ state.content.value.title }}
            </div>
            <div class="hint-body" v-html="state.content.value.body"></div>
        </div>
    </Transition>
</template>

<style scoped>
.hint-panel {
    position: absolute;
    right: 328px;      /* sidebar width (320) + gap (8) */
    top: 100px;
    width: 300px;
    max-height: 70vh;
    overflow-y: auto;
    background: #1e2530;
    border: 1px solid #3a4a5a;
    border-radius: 8px;
    padding: 14px 16px;
    box-shadow: -4px 4px 20px rgba(0, 0, 0, 0.5);
    z-index: 1000;
    pointer-events: none;
    transform: translateY(-50%);
}

.hint-title {
    font-size: 13px;
    font-weight: 700;
    color: #8abcee;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.hint-body {
    font-size: 12px;
    color: #bcc8d8;
    line-height: 1.55;
}

.hint-body :deep(p) {
    margin: 0 0 6px;
}

.hint-body :deep(p:last-child) {
    margin-bottom: 0;
}

.hint-body :deep(code) {
    background: #283040;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 11px;
    color: #99ccff;
    font-family: 'Consolas', 'Fira Mono', monospace;
}

.hint-body :deep(strong) {
    color: #ddeeff;
    font-weight: 600;
}

.hint-body :deep(.hint-values) {
    margin: 6px 0 4px;
    padding-left: 12px;
    border-left: 2px solid #3a4a5a;
    font-size: 11px;
    color: #99aabb;
}

.hint-body :deep(.hint-values div) {
    padding: 1px 0;
}

.hint-body :deep(.hint-note) {
    margin-top: 8px;
    padding: 6px 8px;
    background: #1a2030;
    border-radius: 4px;
    border-left: 3px solid #5588aa;
    font-size: 11px;
    color: #8899aa;
}

/* Transition */
.hint-fade-enter-active { transition: opacity 0.18s ease, transform 0.18s ease; }
.hint-fade-leave-active { transition: opacity 0.12s ease; }
.hint-fade-enter-from { opacity: 0; transform: translateY(-50%) translateX(8px); }
.hint-fade-leave-to   { opacity: 0; }
</style>
