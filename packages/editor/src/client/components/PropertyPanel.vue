<script setup lang="ts">
import { computed } from 'vue';
import { type Engine, NodeType } from '@skengio/engine';
import CollapsibleSection from './CollapsibleSection.vue';
import TransformProps from './props/TransformProps.vue';
import MeshProps from './props/MeshProps.vue';
import LightProps from './props/LightProps.vue';
import CameraProps from './props/CameraProps.vue';
import MaterialProps from './props/MaterialProps.vue';
import AnimationProps from './props/AnimationProps.vue';

const props = defineProps<{
    engine: Engine;
    nodeHandle: number;
}>();

const node = computed(() => props.engine.scene.getNode(props.nodeHandle));

const isInstance = computed(() => (node.value as any)?.isInstance === true);
/** Instance group parent: Empty node that holds mesh/material refs for its children. */
const isInstanceGroup = computed(() =>
    node.value?.type === NodeType.Empty
    && (node.value as any)?.meshHandle !== undefined);
const isMesh = computed(() => node.value?.type === NodeType.Mesh);
const isLight = computed(() => node.value?.type === NodeType.Light);
const isCamera = computed(() => node.value?.type === NodeType.Camera);

const meshHandle = computed(() => (node.value as any)?.meshHandle as number | undefined);
const materialHandles = computed(() => (node.value as any)?.materialHandles as number[] | undefined);
const lightHandle = computed(() => (node.value as any)?.lightHandle as number | undefined);

const hasAnimation = computed(() => {
    if (!node.value) return false;
    return props.engine.animations.getPlaybackState(node.value.handle) !== undefined;
});

const uniqueMaterialHandles = computed(() => {
    const mh = materialHandles.value;
    if (!mh) return [];
    return [...new Set(mh)];
});

function materialLabel(h: number): string {
    return props.engine.materials.getMaterial(h)?.label ?? `Material #${h}`;
}
</script>

<template>
    <div class="property-panel" v-if="node">
        <div class="section-header">
            <i class="fas fa-sliders-h"></i> Properties
        </div>

        <div class="node-info">
            <div class="prop-row">
                <span>Name</span>
                <span class="prop-value name-value">{{ node.label }}</span>
            </div>
            <div class="prop-row">
                <span>Type</span>
                <span class="prop-value">{{ isInstanceGroup ? 'Instance Group' : isInstance ? 'Instance' : node.type }}</span>
            </div>
            <div class="prop-row">
                <span>Handle</span>
                <span class="prop-value dim">{{ node.handle }}</span>
            </div>
            <div v-if="isInstanceGroup" class="prop-row">
                <span>Instances</span>
                <span class="prop-value">{{ node.children.length }}</span>
            </div>
        </div>

        <!-- Transform (all nodes except instance group parents) -->
        <CollapsibleSection v-if="!isInstanceGroup" title="Transform" icon="fas fa-arrows-alt" icon-color="#aaddff">
            <TransformProps :engine="engine" :node-handle="nodeHandle" />
        </CollapsibleSection>

        <!-- Instance children only show transform -->
        <template v-if="!isInstance">
            <!-- Mesh properties -->
            <CollapsibleSection
                v-if="(isMesh || isInstanceGroup) && meshHandle !== undefined"
                title="Mesh"
                icon="fas fa-cube"
                icon-color="#88ddaa"
            >
                <MeshProps
                    :engine="engine"
                    :mesh-handle="meshHandle"
                    :material-handles="materialHandles ?? []"
                />
            </CollapsibleSection>

            <!-- Light properties -->
            <CollapsibleSection
                v-if="isLight && lightHandle !== undefined"
                title="Light"
                icon="fas fa-lightbulb"
                icon-color="#ffd966"
            >
                <LightProps
                    :engine="engine"
                    :light-handle="lightHandle"
                />
            </CollapsibleSection>

            <!-- Camera properties -->
            <CollapsibleSection
                v-if="isCamera"
                title="Camera"
                icon="fas fa-video"
                icon-color="#88bbff"
            >
                <CameraProps :engine="engine" />
            </CollapsibleSection>

            <!-- Animation properties -->
            <CollapsibleSection
                v-if="hasAnimation"
                title="Animation"
                icon="fas fa-film"
                icon-color="#ffaa66"
            >
                <AnimationProps
                    :engine="engine"
                    :node-handle="nodeHandle"
                />
            </CollapsibleSection>

            <!-- Material properties (for mesh nodes, show each unique material) -->
            <CollapsibleSection
                v-for="mh in uniqueMaterialHandles"
                :key="mh"
                :title="materialLabel(mh)"
                icon="fas fa-palette"
                icon-color="#dd88cc"
            >
                <MaterialProps
                    :engine="engine"
                    :material-handle="mh"
                />
            </CollapsibleSection>
        </template>
    </div>
</template>

<style scoped>
.property-panel {
    border-top: 1px solid #3a3a3a;
}

.section-header {
    padding: 8px 14px;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #999;
    background: #2d2d2d;
    display: flex;
    align-items: center;
    gap: 6px;
}

.node-info {
    padding: 8px 14px;
    border-bottom: 1px solid #333;
}

.prop-row {
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
    font-size: 12px;
}

.prop-row span:first-child { color: #888; }
.prop-value { color: #ccc; }
.name-value { font-weight: 500; }
.dim { color: #666; font-size: 10px; font-family: 'Consolas', monospace; }
</style>
