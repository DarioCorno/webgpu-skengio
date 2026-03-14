import { ref, type Ref } from 'vue';

/**
 * Provides a reactive version counter that forces Vue to re-read
 * engine state after mutations. Call `bump()` after any engine edit.
 *
 * Template expressions that access `tick.value` (even as a no-op)
 * will re-evaluate when `bump()` is called.
 */
export function useEngineTick(): { tick: Ref<number>; bump: () => void } {
    const tick = ref(0);
    function bump() { tick.value++; }
    return { tick, bump };
}
