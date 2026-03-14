<script setup lang="ts">
import { computed } from 'vue';
import type { Engine, PlaybackState } from '@skengio/engine';
import NumericInput from '../NumericInput.vue';
import { useEngineTick } from '../../composables/useEngineTick';

const props = defineProps<{
    engine: Engine;
    nodeHandle: number;
}>();

const { tick, bump } = useEngineTick();

const pb = computed(() => {
    tick.value;
    const p = props.engine.animations.getPlaybackState(props.nodeHandle);
    return p ? { ...p } : undefined;
});

const clips = computed(() => props.engine.animations.getClipNames());

function play() {
    if (!pb.value) return;
    props.engine.animations.resume(props.nodeHandle);
    bump();
}

function stop() {
    props.engine.animations.stop(props.nodeHandle);
    bump();
}

function setSpeed(value: number) {
    if (!pb.value) return;
    props.engine.animations.play(props.nodeHandle, {
        clip: pb.value.clipHandle,
        speed: value,
        loop: pb.value.loop,
        autoplay: pb.value.playing,
    });
    bump();
}

function toggleLoop() {
    if (!pb.value) return;
    props.engine.animations.play(props.nodeHandle, {
        clip: pb.value.clipHandle,
        speed: pb.value.speed,
        loop: !pb.value.loop,
        autoplay: pb.value.playing,
    });
    bump();
}
</script>

<template>
    <div v-if="pb">
        <div class="prop-row">
            <span>Clip</span>
            <span class="prop-value">{{ pb!.clipName }}</span>
        </div>

        <div class="prop-row">
            <span>Duration</span>
            <span class="prop-value">{{ pb!.duration.toFixed(2) }}s</span>
        </div>

        <div class="prop-row">
            <span>Time</span>
            <span class="prop-value">{{ pb!.time.toFixed(2) }}s</span>
        </div>

        <div class="prop-label">Speed</div>
        <div class="single-row">
            <NumericInput :model-value="pb!.speed" :step="0.1" :min="0" :max="10" :precision="2" @update:model-value="setSpeed" />
        </div>

        <!-- Playback controls -->
        <div class="controls">
            <button v-if="!pb!.playing" class="ctrl-btn" @click="play" title="Play">
                <i class="fas fa-play"></i>
            </button>
            <button v-else class="ctrl-btn" @click="stop" title="Pause">
                <i class="fas fa-pause"></i>
            </button>
            <button class="ctrl-btn" :class="{ active: pb!.loop }" @click="toggleLoop" title="Loop">
                <i class="fas fa-redo"></i>
            </button>
        </div>

        <!-- Available clips -->
        <div class="prop-label" v-if="clips.length > 1">
            <i class="fas fa-list"></i> Clips ({{ clips.length }})
        </div>
        <div class="clip-list" v-if="clips.length > 1">
            <div class="clip-item" v-for="name in clips" :key="name"
                :class="{ active: name === pb!.clipName }">
                {{ name }}
            </div>
        </div>
    </div>
</template>

<style scoped>
.prop-label { font-size: 11px; color: #888; margin-bottom: 4px; margin-top: 6px; display: flex; align-items: center; gap: 4px; }
.prop-row { display: flex; justify-content: space-between; padding: 2px 0; font-size: 12px; }
.prop-row span:first-child { color: #888; }
.prop-value { color: #ccc; }
.single-row { margin-bottom: 4px; }

.controls { display: flex; gap: 6px; margin-top: 6px; }
.ctrl-btn {
    background: #333;
    border: 1px solid #555;
    border-radius: 4px;
    color: #ccc;
    width: 30px;
    height: 26px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
}
.ctrl-btn:hover { background: #444; border-color: #666; }
.ctrl-btn.active { color: #5b5; border-color: #5b5; }

.clip-list { margin-top: 2px; }
.clip-item { font-size: 11px; color: #777; padding: 1px 4px; border-radius: 2px; }
.clip-item.active { color: #ccc; background: #333; }
</style>
