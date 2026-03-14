import { ref, inject, provide, type Ref } from 'vue';

export interface HintContent {
    title: string;
    body: string;
    /** Optional vertical offset (px from top of sidebar) to anchor near the hovered row. */
    anchorY?: number;
}

export interface HintPanelState {
    visible: Ref<boolean>;
    content: Ref<HintContent | null>;
}

const HINT_PANEL_KEY = Symbol('hintPanel');
const HOVER_DELAY_MS = 600;

/**
 * Call once in App.vue to create the shared state and provide it.
 */
export function provideHintPanel(): HintPanelState {
    const visible = ref(false);
    const content = ref<HintContent | null>(null);
    const state: HintPanelState = { visible, content };
    provide(HINT_PANEL_KEY, state);
    return state;
}

/**
 * Call in any props component to get hover helpers.
 */
export function useHintPanel() {
    const state = inject<HintPanelState>(HINT_PANEL_KEY)!;
    let timer: ReturnType<typeof setTimeout> | null = null;

    function showHint(hint: HintContent) {
        cancelHint();
        timer = setTimeout(() => {
            state.content.value = hint;
            state.visible.value = true;
        }, HOVER_DELAY_MS);
    }

    function cancelHint() {
        if (timer) {
            clearTimeout(timer);
            timer = null;
        }
        state.visible.value = false;
    }

    /**
     * Returns { onMouseenter, onMouseleave } event handlers for a template element.
     * Pass a HintContent or a function that receives the MouseEvent and returns one.
     */
    function hintEvents(hint: HintContent | ((e: MouseEvent) => HintContent)) {
        return {
            mouseenter(e: MouseEvent) {
                const h = typeof hint === 'function' ? hint(e) : { ...hint };
                // Compute anchorY relative to the sidebar
                const sidebar = (e.currentTarget as HTMLElement).closest('.sidebar');
                if (sidebar) {
                    const sidebarRect = sidebar.getBoundingClientRect();
                    const targetRect = (e.currentTarget as HTMLElement).getBoundingClientRect();
                    h.anchorY = targetRect.top - sidebarRect.top + targetRect.height / 2;
                }
                showHint(h);
            },
            mouseleave() {
                cancelHint();
            },
        };
    }

    return { showHint, cancelHint, hintEvents, state };
}
