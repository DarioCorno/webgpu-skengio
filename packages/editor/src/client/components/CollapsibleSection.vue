<script setup lang="ts">
import { ref } from 'vue';

const props = withDefaults(defineProps<{
    title: string;
    icon?: string;
    iconColor?: string;
    collapsed?: boolean;
}>(), {
    icon: '',
    iconColor: '#bbb',
    collapsed: false,
});

const open = ref(!props.collapsed);

function toggle() {
    open.value = !open.value;
}
</script>

<template>
    <div class="collapsible-section">
        <div class="section-header" @click="toggle">
            <i class="fas fa-chevron-right chevron" :class="{ open }"></i>
            <i v-if="icon" :class="icon" :style="{ color: iconColor }"></i>
            <span class="section-title">{{ title }}</span>
        </div>
        <div class="section-body" v-show="open">
            <slot />
        </div>
    </div>
</template>

<style scoped>
.collapsible-section {
    border-bottom: 1px solid #333;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
    color: #bbb;
    cursor: pointer;
    user-select: none;
}

.section-header:hover {
    background: #2a2a2a;
}

.chevron {
    font-size: 9px;
    color: #666;
    transition: transform 0.15s ease;
    width: 10px;
    text-align: center;
}

.chevron.open {
    transform: rotate(90deg);
}

.section-title {
    flex: 1;
}

.section-body {
    padding: 4px 14px 8px;
}
</style>
