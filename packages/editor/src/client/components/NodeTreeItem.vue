<script setup lang="ts">
import { NodeType } from '@skengio/engine';

interface TreeNode {
    handle: number;
    label: string;
    type: NodeType;
    children: TreeNode[];
    expanded: boolean;
    isRoot?: boolean;
    isInstance?: boolean;
}

const props = defineProps<{
    node: TreeNode;
    selectedNode: number | null;
    depth: number;
}>();

const emit = defineEmits<{
    'select-node': [handle: number | null];
    'toggle': [handle: number];
}>();

function toggle() {
    emit('toggle', props.node.handle);
}

/** True when this node is an instance group parent (Empty with instance children). */
function isInstanceGroup(node: TreeNode): boolean {
    return node.type === NodeType.Empty && node.children.length > 0
        && node.children[0]!.isInstance === true;
}

function nodeIcon(node: TreeNode): string {
    if (node.isRoot) return 'fa-film';
    if (node.isInstance) return 'fa-clone';
    if (isInstanceGroup(node)) return 'fa-layer-group';
    switch (node.type) {
        case NodeType.Mesh: return 'fa-cube';
        case NodeType.Light: return 'fa-lightbulb';
        case NodeType.Camera: return 'fa-video';
        default: return 'fa-circle-dot';
    }
}

function nodeIconColor(node: TreeNode): string {
    if (node.isRoot) return '#8cb4ff';
    if (node.isInstance) return '#8eaacc';
    if (isInstanceGroup(node)) return '#66ccaa';
    switch (node.type) {
        case NodeType.Mesh: return '#6cb4ee';
        case NodeType.Light: return '#ffd966';
        case NodeType.Camera: return '#88bbff';
        default: return '#aaa';
    }
}
</script>

<template>
    <div>
        <div
            class="tree-node"
            :class="{ selected: node.handle === selectedNode, root: node.isRoot }"
            :style="{ paddingLeft: (14 + depth * 16) + 'px' }"
            @click.stop="emit('select-node', node.handle)"
        >
            <span
                v-if="!node.isRoot && node.children.length"
                class="expand-toggle"
                @click.stop="toggle"
            >
                <i :class="node.expanded ? 'fas fa-caret-down' : 'fas fa-caret-right'"></i>
            </span>
            <span v-else-if="!node.isRoot" class="expand-spacer"></span>
            <i class="fas node-icon" :class="nodeIcon(node)" :style="{ color: nodeIconColor(node) }"></i>
            <span class="node-label" :class="{ 'root-label': node.isRoot, 'instance-label': node.isInstance }">{{ node.label }}</span>
            <span v-if="isInstanceGroup(node)" class="instance-count">{{ node.children.length }}</span>
        </div>
        <template v-if="node.expanded && node.children.length">
            <NodeTreeItem
                v-for="child in node.children"
                :key="child.handle"
                :node="child"
                :selected-node="selectedNode"
                :depth="depth + 1"
                @select-node="(h) => emit('select-node', h)"
                @toggle="(h) => emit('toggle', h)"
            />
        </template>
    </div>
</template>

<script lang="ts">
export default { name: 'NodeTreeItem' };
</script>

<style scoped>
.tree-node {
    display: flex;
    align-items: center;
    padding: 4px 14px;
    cursor: pointer;
    gap: 4px;
}
.tree-node:hover { background: #333; }
.tree-node.selected { background: #2a4a7a; }
.tree-node.root:hover { background: #2a2a2a; }

.expand-toggle {
    width: 16px;
    text-align: center;
    color: #888;
    font-size: 11px;
    flex-shrink: 0;
}
.expand-spacer { width: 16px; flex-shrink: 0; }

.node-icon {
    font-size: 12px;
    width: 16px;
    text-align: center;
    flex-shrink: 0;
}

.node-label {
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.root-label {
    font-weight: 600;
    color: #ccc;
}

.instance-label {
    color: #888;
    font-style: italic;
}

.instance-count {
    margin-left: auto;
    font-size: 10px;
    color: #666;
    background: #333;
    padding: 1px 5px;
    border-radius: 8px;
    flex-shrink: 0;
}
</style>
