<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { type Engine, NodeType } from '@skengio/engine';
import NodeTreeItem from './NodeTreeItem.vue';

const props = defineProps<{
    engine: Engine;
    sceneName: string;
    selectedNode: number | null;
}>();

const emit = defineEmits<{
    'select-node': [handle: number | null];
}>();

interface TreeNode {
    handle: number;
    label: string;
    type: NodeType;
    children: TreeNode[];
    expanded: boolean;
    isRoot?: boolean;
    isInstance?: boolean;
}

const root = ref<TreeNode | null>(null);
// Track which handles the user has explicitly toggled
// Stores handles that have been explicitly expanded or collapsed by the user
const expandedOverrides = new Map<number, boolean>();

function buildTree(): TreeNode {
    const scene = props.engine.scene;
    const nodeMap = new Map<number, TreeNode>();
    const roots: TreeNode[] = [];

    // Collect handles of nodes that have instance children so we can
    // default-collapse them (they can contain hundreds of items).
    const instanceGroupHandles = new Set<number>();

    for (const node of (scene as any)._nodes.values()) {
        if (node.isInstance && node.parent != null) {
            instanceGroupHandles.add(node.parent);
        }
    }

    for (const node of (scene as any)._nodes.values()) {
        const isGroup = instanceGroupHandles.has(node.handle);
        nodeMap.set(node.handle, {
            handle: node.handle,
            label: node.label || `Node ${node.handle}`,
            type: node.type,
            children: [],
            // Instance groups default to collapsed (can have hundreds of items)
            expanded: expandedOverrides.has(node.handle)
                ? expandedOverrides.get(node.handle)!
                : !isGroup,
            isInstance: node.isInstance ?? false,
        });
    }

    for (const node of (scene as any)._nodes.values()) {
        const treeNode = nodeMap.get(node.handle)!;
        if (node.parent != null && nodeMap.has(node.parent)) {
            nodeMap.get(node.parent)!.children.push(treeNode);
        } else {
            roots.push(treeNode);
        }
    }

    // Track top-level handles and default them to collapsed
    topLevelHandles.clear();
    for (const r of roots) {
        topLevelHandles.add(r.handle);
        if (!expandedOverrides.has(r.handle)) {
            r.expanded = false;
        }
    }

    return {
        handle: -1,
        label: props.sceneName,
        type: NodeType.Empty,
        children: roots,
        expanded: true,
        isRoot: true,
    };
}

function refreshTree() {
    root.value = buildTree();
}

// Track which handles are top-level scene children (for default-collapsed logic)
const topLevelHandles = new Set<number>();

function onToggle(handle: number) {
    // Root node is always expanded, ignore toggle
    if (handle === -1) return;
    if (expandedOverrides.has(handle)) {
        expandedOverrides.set(handle, !expandedOverrides.get(handle));
    } else {
        // First toggle: top-level nodes default collapsed, others default expanded
        const defaultExpanded = !topLevelHandles.has(handle);
        expandedOverrides.set(handle, !defaultExpanded);
    }
    refreshTree();
}

onMounted(() => {
    refreshTree();
    setInterval(refreshTree, 2000);
});
</script>

<template>
    <div class="node-tree">
        <div class="section-header" @click="refreshTree">
            <i class="fas fa-sitemap"></i> Scene Graph
            <button class="refresh-btn" title="Refresh">
                <i class="fas fa-sync-alt"></i>
            </button>
        </div>
        <div class="tree-content" v-if="root">
            <NodeTreeItem
                :node="root"
                :selected-node="selectedNode"
                :depth="0"
                @select-node="(h) => emit('select-node', h)"
                @toggle="onToggle"
            />
        </div>
    </div>
</template>

<style scoped>
.node-tree {
    border-bottom: 1px solid #3a3a3a;
    max-height: 40%;
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
}

.section-header {
    padding: 8px 14px;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #999;
    background: #2d2d2d;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
}

.refresh-btn {
    margin-left: auto;
    background: none;
    border: none;
    color: #666;
    cursor: pointer;
    font-size: 11px;
    padding: 2px 4px;
}
.refresh-btn:hover { color: #aaa; }

.tree-content {
    padding: 4px 0;
    overflow-y: auto;
    flex: 1;
    min-height: 0;
}
</style>
