<script setup lang="ts">
import type { Engine } from '@skengio/engine';
import NodeTree from './NodeTree.vue';
import PropertyPanel from './PropertyPanel.vue';
import EngineConfigProps from './props/EngineConfigProps.vue';

defineProps<{
    engine: Engine;
    sceneName: string;
    selectedNode: number | null;
}>();

const emit = defineEmits<{
    'select-node': [handle: number | null];
}>();
</script>

<template>
    <aside class="sidebar">
        <div class="sidebar-content">
            <NodeTree
                :engine="engine"
                :scene-name="sceneName"
                :selected-node="selectedNode"
                @select-node="(h) => emit('select-node', h)"
            />
            <PropertyPanel
                v-if="selectedNode !== null && selectedNode !== -1"
                :engine="engine"
                :node-handle="selectedNode"
            />
            <EngineConfigProps v-if="selectedNode === -1" :engine="engine" />
        </div>
    </aside>
</template>

<style scoped>
.sidebar {
    width: 320px;
    min-width: 320px;
    height: 100%;
    background: #252525;
    border-left: 1px solid #3a3a3a;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.sidebar-content {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
}
</style>
