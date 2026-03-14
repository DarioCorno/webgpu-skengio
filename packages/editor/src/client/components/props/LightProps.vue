<script setup lang="ts">
import { computed } from 'vue';
import { type Engine, type LightRecord, LightType } from '@skengio/engine';
import NumericInput from '../NumericInput.vue';
import ColorSwatch from '../ColorSwatch.vue';
import { useEngineTick } from '../../composables/useEngineTick';
import { useHintPanel } from '../../composables/useHintPanel';
import { lightHints } from '../../hints/lightHints';

const props = defineProps<{
    engine: Engine;
    lightHandle: number;
}>();

const { tick, bump } = useEngineTick();
const { hintEvents } = useHintPanel();
const hint = (key: string) => hintEvents(lightHints[key]!);

const light = computed(() => {
    tick.value;
    const l = props.engine.lights.getLight(props.lightHandle);
    return l ? { ...l } : undefined;
});

function update(changes: Record<string, unknown>) {
    props.engine.lights.updateLight(props.lightHandle, changes as any);
    bump();
}

function updateColor(axis: number, value: number) {
    const l = light.value;
    if (!l) return;
    const c: [number, number, number] = [...l.color];
    c[axis] = Math.max(0, Math.min(1, value));
    update({ color: c });
}

function onSwatchUpdate(r: number, g: number, b: number) {
    update({ color: [r, g, b] });
}

function isSpot(l: LightRecord) { return l.type === LightType.Spot; }
function isPoint(l: LightRecord) { return l.type === LightType.Point || l.type === LightType.Spot; }
</script>

<template>
    <div v-if="light">
        <div class="prop-row" v-on="hint('type')">
            <span>Type</span>
            <span class="prop-value">{{ light!.type }}</span>
        </div>

        <!-- Color -->
        <div class="prop-label" v-on="hint('color')">Color (RGB)</div>
        <div class="color-preview-row">
            <ColorSwatch
                :r="light!.color[0]"
                :g="light!.color[1]"
                :b="light!.color[2]"
                @update:rgb="onSwatchUpdate"
            />
            <div class="color-inputs">
                <div class="vec3-row">
                    <NumericInput label="R" :model-value="light!.color[0]" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => updateColor(0, v)" />
                    <NumericInput label="G" :model-value="light!.color[1]" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => updateColor(1, v)" />
                    <NumericInput label="B" :model-value="light!.color[2]" :step="0.01" :min="0" :max="1" :precision="3" @update:model-value="v => updateColor(2, v)" />
                </div>
            </div>
        </div>

        <!-- Intensity -->
        <div class="prop-label" v-on="hint('intensity')">Intensity</div>
        <div class="single-row">
            <NumericInput :model-value="light!.intensity" :step="0.1" :min="0" @update:model-value="v => update({ intensity: v })" />
        </div>

        <!-- Range (point/spot) -->
        <template v-if="isPoint(light!)">
            <div class="prop-label" v-on="hint('range')">Range</div>
            <div class="single-row">
                <NumericInput :model-value="light!.range" :step="0.5" :min="0.1" @update:model-value="v => update({ range: v })" />
            </div>
        </template>

        <!-- Cone angles (spot) -->
        <template v-if="isSpot(light!)">
            <div class="prop-label" v-on="hint('innerCone')">Inner Cone (rad)</div>
            <div class="single-row">
                <NumericInput :model-value="light!.innerConeAngle" :step="0.01" :min="0" :max="1.57" :precision="3" @update:model-value="v => update({ innerConeAngle: v })" />
            </div>
            <div class="prop-label" v-on="hint('outerCone')">Outer Cone (rad)</div>
            <div class="single-row">
                <NumericInput :model-value="light!.outerConeAngle" :step="0.01" :min="0" :max="1.57" :precision="3" @update:model-value="v => update({ outerConeAngle: v })" />
            </div>
        </template>

        <!-- Shadow -->
        <div class="prop-row" v-on="hint('castShadow')">
            <span>Cast Shadow</span>
            <span class="prop-value">{{ light!.castShadow ? 'Yes' : 'No' }}</span>
        </div>
        <template v-if="light!.castShadow">
            <div class="prop-label" v-on="hint('shadowBias')">Shadow Bias</div>
            <div class="single-row">
                <NumericInput :model-value="light!.shadowBias" :step="0.001" :min="0" :precision="4" @update:model-value="v => update({ shadowBias: v })" />
            </div>
            <div class="prop-label" v-on="hint('pcfRadius')">PCF Radius</div>
            <div class="single-row">
                <NumericInput :model-value="light!.pcfRadius" :step="1" :min="0" :max="3" :precision="0" @update:model-value="v => update({ pcfRadius: v })" />
            </div>
            <div class="prop-row" v-on="hint('shadowType')">
                <span>Shadow Type</span>
                <span class="prop-value">{{ ['None', 'Standard', 'Cascaded', 'Cube'][light!.shadowType] }}</span>
            </div>
            <div class="prop-row" v-on="hint('shadowResolution')">
                <span>Resolution</span>
                <span class="prop-value">{{ light!.shadowMapResolution }}px</span>
            </div>
            <template v-if="light!.numCascades > 1">
                <div class="prop-row" v-on="hint('cascades')">
                    <span>Cascades</span>
                    <span class="prop-value">{{ light!.numCascades }}</span>
                </div>
            </template>
        </template>
    </div>
</template>

<style scoped>
.prop-label { font-size: 11px; color: #888; margin-bottom: 4px; margin-top: 6px; display: flex; align-items: center; gap: 4px; cursor: help; }
.prop-row { display: flex; justify-content: space-between; padding: 2px 0; font-size: 12px; }
.prop-row span:first-child { color: #888; }
.prop-value { color: #ccc; }
.vec3-row { display: flex; gap: 6px; margin-bottom: 4px; }
.vec3-row > * { flex: 1; min-width: 0; }
.single-row { margin-bottom: 4px; }
.color-preview-row { display: flex; gap: 8px; align-items: flex-start; }
.color-inputs { flex: 1; min-width: 0; }
</style>
