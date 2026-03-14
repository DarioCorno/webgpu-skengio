<script setup lang="ts">
import { ref } from 'vue';
import AlertDialog from './AlertDialog.vue';

const emit = defineEmits<{
    'open-scene': [file: string];
    'save-scene': [];
    'fullscreen': [];
}>();

const openMenu = ref<string | null>(null);
const hoveredItem = ref<string | null>(null);
const sceneFiles = ref<string[]>([]);

function toggleMenu(name: string) {
    openMenu.value = openMenu.value === name ? null : name;
    if (openMenu.value === 'file') {
        fetchScenes();
    }
}

function closeMenu() {
    openMenu.value = null;
    hoveredItem.value = null;
}

async function fetchScenes() {
    try {
        const res = await fetch('/api/scenes');
        sceneFiles.value = await res.json();
    } catch {
        sceneFiles.value = [];
    }
}

function onNew() {
    closeMenu();
    console.log('[Menu] New');
}

function onOpenScene(file: string) {
    closeMenu();
    emit('open-scene', file);
}

const saveConfirmOpen = ref(false);

function onSave() {
    closeMenu();
    saveConfirmOpen.value = true;
}

function onSaveConfirm() {
    saveConfirmOpen.value = false;
    emit('save-scene');
}

function onSaveCancel() {
    saveConfirmOpen.value = false;
}

function onClose() {
    closeMenu();
    console.log('[Menu] Close');
}

// View menu handlers
function onFullscreen() {
    closeMenu();
    emit('fullscreen');
}

function onRun() {
    closeMenu();
    console.log('[Menu] Run');
}
</script>

<template>
    <div class="menubar" @mouseleave="closeMenu">
        <div class="menu-root" :class="{ active: openMenu === 'file' }" @click="toggleMenu('file')" @mouseenter="openMenu && (openMenu = 'file')">
            File
            <div class="dropdown" v-if="openMenu === 'file'">
                <div class="menu-item" @click.stop="onNew">New</div>
                <div
                    class="menu-item has-sub"
                    @mouseenter="hoveredItem = 'open'"
                    @mouseleave="hoveredItem = null"
                    @click.stop
                >
                    Open
                    <i class="fas fa-chevron-right sub-arrow"></i>
                    <div class="submenu" v-if="hoveredItem === 'open'">
                        <div
                            v-for="file in sceneFiles"
                            :key="file"
                            class="menu-item"
                            @click.stop="onOpenScene(file)"
                        >{{ file.replace(/\.json$/, '') }}</div>
                        <div v-if="!sceneFiles.length" class="menu-item disabled">No scenes found</div>
                    </div>
                </div>
                <div class="menu-item" @click.stop="onSave"><i class="fas fa-save menu-icon"></i>Save</div>
                <div class="separator"></div>
                <div class="menu-item" @click.stop="onClose">Close</div>
            </div>
        </div>
        <div class="menu-root" :class="{ active: openMenu === 'view' }" @click="toggleMenu('view')" @mouseenter="openMenu && (openMenu = 'view')">
            View
            <div class="dropdown" v-if="openMenu === 'view'">
                <div class="menu-item" @click.stop="onFullscreen"><i class="fas fa-expand menu-icon"></i>Fullscreen</div>
                <div class="menu-item" @click.stop="onRun"><i class="fas fa-play menu-icon"></i>Run</div>
            </div>
        </div>
    </div>
    <AlertDialog
        v-if="saveConfirmOpen"
        icon="fas fa-save"
        title="Save Scene"
        body="This will overwrite the current scene file. Are you sure?"
        confirm-text="Save"
        confirm-icon="fas fa-save"
        cancel-text="Cancel"
        cancel-icon="fas fa-times"
        @confirm="onSaveConfirm"
        @cancel="onSaveCancel"
    />
</template>

<style scoped>
.menubar {
    display: flex;
    align-items: stretch;
    background: #2b2b2b;
    border-bottom: 1px solid #1a1a1a;
    height: 28px;
    flex-shrink: 0;
    font-size: 12px;
    user-select: none;
    z-index: 100;
}

.menu-root {
    position: relative;
    display: flex;
    align-items: center;
    padding: 0 10px;
    color: #ccc;
    cursor: pointer;
}

.menu-root:hover,
.menu-root.active {
    background: #3c3c3c;
}

.dropdown {
    position: absolute;
    top: 100%;
    left: 0;
    min-width: 160px;
    background: #2b2b2b;
    border: 1px solid #1a1a1a;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    padding: 4px 0;
    z-index: 200;
}

.menu-icon {
    width: 16px;
    margin-right: 8px;
    font-size: 11px;
    text-align: center;
    color: #999;
}

.menu-item:hover .menu-icon {
    color: #fff;
}

.menu-item {
    padding: 5px 24px;
    color: #ccc;
    cursor: pointer;
    white-space: nowrap;
    position: relative;
}

.menu-item:hover {
    background: #094771;
    color: #fff;
}

.menu-item.disabled {
    color: #666;
    cursor: default;
}

.menu-item.disabled:hover {
    background: transparent;
    color: #666;
}

.has-sub {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.sub-arrow {
    font-size: 8px;
    color: #777;
    margin-left: 12px;
}

.submenu {
    position: absolute;
    left: 100%;
    top: -4px;
    min-width: 180px;
    background: #2b2b2b;
    border: 1px solid #1a1a1a;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    padding: 4px 0;
    z-index: 300;
}

.separator {
    height: 1px;
    background: #3c3c3c;
    margin: 4px 0;
}
</style>
