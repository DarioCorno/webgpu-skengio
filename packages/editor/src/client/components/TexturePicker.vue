<script setup lang="ts">
import { inject, ref, computed } from 'vue';
import type { Ref } from 'vue';

const props = defineProps<{
    slot: string;
    files?: string[];
}>();

const emit = defineEmits<{
    pick: [file: string];
    clear: [];
    close: [];
}>();

const textureFiles = inject<Ref<string[]>>('textureFiles', ref([]));

const filter = ref('');

const sourceFiles = computed(() => props.files ?? textureFiles.value);

const filteredFiles = computed(() => {
    const q = filter.value.toLowerCase();
    if (!q) return sourceFiles.value;
    return sourceFiles.value.filter(f => f.toLowerCase().includes(q));
});

function isCubemap(file: string): boolean {
    return file.startsWith('cubemaps/');
}

function displayName(file: string): string {
    // Show just the filename or cubemap folder name
    const parts = file.split('/');
    return parts[parts.length - 1];
}

function previewUrl(file: string): string {
    if (isCubemap(file)) {
        // Show one face as preview
        return `/textures/${file}/posx.jpg`;
    }
    return `/textures/${file}`;
}

const refreshing = ref(false);
const uploading = ref(false);
const fileInputRef = ref<HTMLInputElement | null>(null);

async function refreshTextures() {
    refreshing.value = true;
    try {
        const res = await fetch('/api/textures');
        textureFiles.value = await res.json();
    } catch { /* ignore */ }
    refreshing.value = false;
}

function triggerUpload() {
    fileInputRef.value?.click();
}

async function handleFileUpload(event: Event) {
    const input = event.target as HTMLInputElement;
    const files = input.files;
    if (!files || files.length === 0) return;

    uploading.value = true;
    try {
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('textures', files[i]);
        }
        const res = await fetch('/api/textures/upload', {
            method: 'POST',
            body: formData,
        });
        if (res.ok) {
            await refreshTextures();
        }
    } catch { /* ignore */ }
    uploading.value = false;
    // Reset input so the same file can be re-selected
    input.value = '';
}
</script>

<template>
    <Teleport to="body">
        <div class="picker-overlay" @click.self="emit('close')">
            <div class="picker-modal">
                <div class="picker-header">
                    <span class="picker-title">
                        <i class="fas fa-image"></i> Select Texture — {{ slot }}
                    </span>
                    <button class="close-btn" @click="emit('close')">
                        <i class="fas fa-times"></i>
                    </button>
                </div>

                <div class="picker-toolbar">
                    <input
                        v-model="filter"
                        class="filter-input"
                        type="text"
                        placeholder="Filter textures..."
                    />
                    <button class="upload-tex-btn" @click="triggerUpload" title="Upload texture(s)" :disabled="uploading">
                        <i class="fas" :class="uploading ? 'fa-spinner fa-spin' : 'fa-upload'"></i> Upload
                    </button>
                    <input
                        ref="fileInputRef"
                        type="file"
                        accept=".jpg,.jpeg,.png,.webp,.hdr,.exr,.bmp,.tga"
                        multiple
                        style="display: none"
                        @change="handleFileUpload"
                    />
                    <button class="refresh-tex-btn" @click="refreshTextures" title="Refresh texture list" :disabled="refreshing">
                        <i class="fas fa-sync-alt" :class="{ 'fa-spin': refreshing }"></i>
                    </button>
                    <button class="clear-tex-btn" @click="emit('clear')" title="Remove texture">
                        <i class="fas fa-trash-alt"></i> None
                    </button>
                </div>

                <div class="picker-grid">
                    <div
                        v-for="file in filteredFiles"
                        :key="file"
                        class="tex-card"
                        @click="emit('pick', file)"
                    >
                        <div class="tex-thumb">
                            <img
                                :src="previewUrl(file)"
                                :alt="file"
                                loading="lazy"
                                @error="($event.target as HTMLImageElement).style.display = 'none'"
                            />
                            <i v-if="isCubemap(file)" class="fas fa-cube cubemap-badge"></i>
                            <div class="tex-name-overlay">{{ displayName(file) }}</div>
                        </div>
                    </div>
                    <div v-if="filteredFiles.length === 0" class="no-results">
                        No textures found
                    </div>
                </div>
            </div>
        </div>
    </Teleport>
</template>

<style scoped>
.picker-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    z-index: 500;
    display: flex;
    align-items: center;
    justify-content: center;
}

.picker-modal {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 8px;
    width: 720px;
    max-width: 90vw;
    max-height: 85vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
}

.picker-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    border-bottom: 1px solid #3a3a3a;
}

.picker-title {
    font-size: 13px;
    font-weight: 600;
    color: #ccc;
    display: flex;
    align-items: center;
    gap: 6px;
}

.close-btn {
    background: none;
    border: none;
    color: #888;
    cursor: pointer;
    font-size: 14px;
    padding: 4px;
}
.close-btn:hover { color: #ddd; }

.picker-toolbar {
    display: flex;
    gap: 8px;
    padding: 8px 14px;
    border-bottom: 1px solid #333;
}

.filter-input {
    flex: 1;
    background: #1e1e1e;
    border: 1px solid #444;
    border-radius: 4px;
    color: #ccc;
    font-size: 12px;
    padding: 4px 8px;
    outline: none;
}
.filter-input:focus { border-color: #5588cc; }

.upload-tex-btn {
    background: #1e3a1e;
    border: 1px solid #3a6633;
    border-radius: 4px;
    color: #88cc88;
    font-size: 11px;
    padding: 4px 10px;
    cursor: pointer;
    white-space: nowrap;
}
.upload-tex-btn:hover { background: #2a4a2a; color: #aaddaa; }
.upload-tex-btn:disabled { opacity: 0.5; cursor: default; }

.refresh-tex-btn {
    background: #2a2a3a;
    border: 1px solid #555;
    border-radius: 4px;
    color: #999;
    font-size: 12px;
    padding: 4px 8px;
    cursor: pointer;
}
.refresh-tex-btn:hover { background: #333; color: #ccc; }
.refresh-tex-btn:disabled { opacity: 0.5; cursor: default; }

.clear-tex-btn {
    background: #3a2020;
    border: 1px solid #663333;
    border-radius: 4px;
    color: #cc8888;
    font-size: 11px;
    padding: 4px 10px;
    cursor: pointer;
    white-space: nowrap;
}
.clear-tex-btn:hover { background: #4a2a2a; color: #eaa; }

.picker-grid {
    flex: 1;
    overflow-y: auto;
    padding: 10px 14px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
    gap: 8px;
    align-content: start;
}

.tex-card {
    cursor: pointer;
    border-radius: 4px;
    background: #222;
    border: 1px solid #3a3a3a;
    transition: border-color 0.15s;
}
.tex-card:hover {
    border-color: #5588cc;
}

.tex-thumb {
    width: 100%;
    aspect-ratio: 1;
    background: #1a1a1a;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}

.tex-thumb img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.cubemap-badge {
    position: absolute;
    bottom: 3px;
    right: 3px;
    font-size: 10px;
    color: #88bbff;
    background: rgba(0,0,0,0.6);
    padding: 2px 3px;
    border-radius: 2px;
}

.tex-name-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(0, 0, 0, 0.72);
    color: #ddd;
    font-size: 9px;
    padding: 3px 4px;
    text-align: center;
    word-break: break-all;
    line-height: 1.3;
}

.no-results {
    grid-column: 1 / -1;
    text-align: center;
    color: #666;
    font-size: 12px;
    padding: 20px;
}
</style>
