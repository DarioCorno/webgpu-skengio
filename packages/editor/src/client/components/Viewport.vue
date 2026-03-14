<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue';
import { Engine, SceneLoader } from '@skengio/engine';

const props = defineProps<{
    defaultScene: string;
}>();

const emit = defineEmits<{
    'engine-ready': [engine: Engine];
    'scene-loaded': [sceneData: Record<string, any>];
}>();

const canvasRef = ref<HTMLCanvasElement | null>(null);
let engine: Engine | null = null;
let loader: SceneLoader | null = null;

async function loadScene(url: string) {
    if (!loader || !engine) return;
    const res = await fetch(url);
    const json = await res.text();
    await loader.loadFromString(json);
    emit('scene-loaded', JSON.parse(json));
}

defineExpose({ loadScene });

onMounted(async () => {
    const canvas = canvasRef.value!;
    canvas.width = Math.floor(canvas.clientWidth * devicePixelRatio);
    canvas.height = Math.floor(canvas.clientHeight * devicePixelRatio);

    engine = new Engine();
    await engine.init({ canvas });

    // Load default scene
    loader = new SceneLoader();
    const defaultUrl = `/scenes/${props.defaultScene}.json`;
    await loader.load(engine, defaultUrl, canvas);

    // Fetch scene JSON for the editor to reference
    let sceneData: Record<string, any> = {};
    try {
        const res = await fetch(defaultUrl);
        sceneData = await res.json();
    } catch { /* ignore */ }

    engine.start();
    emit('engine-ready', engine);
    emit('scene-loaded', sceneData);
});

onBeforeUnmount(() => {
    if (engine) {
        engine.stop();
        engine.destroy();
        engine = null;
    }
});
</script>

<template>
    <canvas ref="canvasRef" class="viewport-canvas" />
</template>

<style scoped>
.viewport-canvas {
    flex: 1;
    min-width: 0;
    display: block;
    height: 100%;
}
</style>
