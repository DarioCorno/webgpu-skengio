<script setup lang="ts">
const props = withDefaults(defineProps<{
    icon?: string;
    title: string;
    body?: string;
    confirmText?: string;
    cancelText?: string;
    showConfirm?: boolean;
    showCancel?: boolean;
    confirmIcon?: string;
    cancelIcon?: string;
}>(), {
    showConfirm: true,
    showCancel: true,
});

const emit = defineEmits<{
    confirm: [];
    cancel: [];
}>();
</script>

<template>
    <Teleport to="body">
        <div class="alert-overlay">
            <div class="alert-modal">
                <div class="alert-header">
                    <i v-if="icon" :class="icon" class="alert-icon"></i>
                    <span class="alert-title">{{ title }}</span>
                </div>
                <div v-if="body" class="alert-body">{{ body }}</div>
                <div class="alert-actions">
                    <button
                        v-if="showCancel"
                        class="alert-btn cancel"
                        @click="emit('cancel')"
                    ><i v-if="cancelIcon" :class="cancelIcon" class="btn-icon"></i>{{ cancelText ?? 'Cancel' }}</button>
                    <button
                        v-if="showConfirm"
                        class="alert-btn confirm"
                        @click="emit('confirm')"
                    ><i v-if="confirmIcon" :class="confirmIcon" class="btn-icon"></i>{{ confirmText ?? 'OK' }}</button>
                </div>
            </div>
        </div>
    </Teleport>
</template>

<style scoped>
.alert-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.55);
    z-index: 600;
    display: flex;
    align-items: center;
    justify-content: center;
}

.alert-modal {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 8px;
    width: 380px;
    max-width: 90vw;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    display: flex;
    flex-direction: column;
}

.alert-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 14px 16px 0;
}

.alert-icon {
    font-size: 16px;
    color: #88aacc;
    flex-shrink: 0;
}

.alert-title {
    font-size: 14px;
    font-weight: 600;
    color: #ddd;
}

.alert-body {
    padding: 10px 16px 0;
    font-size: 12px;
    color: #999;
    line-height: 1.5;
    white-space: pre-wrap;
}

.alert-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 14px 16px;
}

.alert-btn {
    border: 1px solid #555;
    border-radius: 4px;
    font-size: 12px;
    padding: 5px 16px;
    cursor: pointer;
    min-width: 70px;
}

.alert-btn.cancel {
    background: #333;
    color: #aaa;
}
.alert-btn.cancel:hover {
    background: #3a3a3a;
    color: #ccc;
}

.alert-btn.confirm {
    background: #2a4a7a;
    border-color: #5588cc;
    color: #ddd;
}
.alert-btn.confirm:hover {
    background: #335a8a;
    color: #fff;
}

.btn-icon {
    margin-right: 6px;
    font-size: 11px;
}
</style>
