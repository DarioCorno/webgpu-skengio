// /src/engine/input/InputSystem.ts
//
// Collects raw keyboard, mouse input each frame.
//
// Typical usage (one call per frame):
//   inputSystem.beginFrame();          // resets per-frame deltas
//   const state = inputSystem.getState();
//   // … pass state to camera controllers, UI, game logic …

// ─────────────────────────────────────────────────────────────────────────────
// Public read-only interfaces consumed by camera controllers and game code
// ─────────────────────────────────────────────────────────────────────────────

export interface KeyboardState {
    /** Set of currently-pressed key codes (e.g. "KeyW", "ShiftLeft"). */
    readonly keys: ReadonlySet<string>;
    /** True while the key identified by `code` is held down. */
    isDown(code: string): boolean;
}

export interface MouseState {
    /** Pointer delta accumulated since the last beginFrame() call (pixels). */
    readonly dx: number;
    readonly dy: number;
    /** Current screen-space cursor position (pixels). */
    readonly x: number;
    readonly y: number;
    /** Bitmask of pressed buttons: bit 0 = LMB, bit 1 = MMB, bit 2 = RMB. */
    readonly buttons: number;
    /** True when the Pointer Lock API has captured the cursor. */
    readonly locked: boolean;
    /** Scroll wheel delta accumulated since the last beginFrame() call. Positive = scroll down. */
    readonly scrollDelta: number;
    isButtonDown(button: number): boolean;
}

export interface GamepadState {
    /** True when at least one gamepad is connected and readable. */
    readonly connected: boolean;
    /**
     * Axis values after deadzone.  Standard mapping:
     *   0 = left stick X (-1 left, +1 right)
     *   1 = left stick Y (-1 up, +1 down)
     *   2 = right stick X
     *   3 = right stick Y
     */
    readonly axes: readonly number[];
    /** Button pressed states (standard mapping indices). */
    readonly buttons: readonly boolean[];
}

export interface InputState {
    readonly keyboard: KeyboardState;
    readonly mouse:    MouseState;
    readonly gamepad:  GamepadState;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal mutable implementations
// ─────────────────────────────────────────────────────────────────────────────

class KeyboardStateImpl implements KeyboardState {
    readonly keys = new Set<string>();
    isDown(code: string): boolean { return this.keys.has(code); }
}

class MouseStateImpl implements MouseState {
    dx = 0; dy = 0;
    x  = 0; y  = 0;
    buttons = 0;
    locked  = false;
    scrollDelta = 0;
    isButtonDown(button: number): boolean { return (this.buttons & (1 << button)) !== 0; }
}

const GAMEPAD_DEADZONE = 0.15;

class GamepadStateImpl implements GamepadState {
    connected = false;
    axes: number[] = [0, 0, 0, 0];
    buttons: boolean[] = [];
}

// ─────────────────────────────────────────────────────────────────────────────
// InputSystem
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Manages raw input from keyboard, mouse
 *
 * Call `init(canvas)` once, then `beginFrame()` + `getState()` each frame.
 *
 * Pointer lock is requested when the user clicks the canvas.
 * Press Escape (browser default) to release the lock.
 */
export class InputSystem {

    private _enabled = true;

    private readonly _keyboard = new KeyboardStateImpl();
    private readonly _mouse    = new MouseStateImpl();
    private readonly _gamepad  = new GamepadStateImpl();

    // Raw mouse-movement accumulators written to by DOM events.
    // Snapshotted into _mouse.dx/dy at beginFrame() so the controller
    // reads the delta accumulated between the previous frame and this one,
    // not a zero that was just wiped by beginFrame() itself.
    private _rawMouseDx = 0;
    private _rawMouseDy = 0;
    private _rawScrollDelta = 0;

    /**
     * True while any mouse button was pressed on the canvas and not yet released.
     * Used to continue accumulating mouse deltas during drag even when the
     * cursor leaves the canvas (e.g. orbit / pan controllers without pointer lock).
     */
    private _dragging = false;

    /**
     * When true (default), clicking the canvas requests pointer lock.
     * Set to false for controllers (e.g. orbit/editor) that use click-drag instead.
     */
    pointerLockEnabled = true;

    private _canvas: HTMLCanvasElement | null = null;

    // Stored handler refs so they can be removed with removeEventListener
    private _hKeyDown:           (e: KeyboardEvent) => void  = () => {};
    private _hKeyUp:             (e: KeyboardEvent) => void  = () => {};
    private _hMouseMove:         (e: MouseEvent)    => void  = () => {};
    private _hMouseDown:         (e: MouseEvent)    => void  = () => {};
    private _hMouseUp:           (e: MouseEvent)    => void  = () => {};
    private _hWheel:             (e: WheelEvent)    => void  = () => {};
    private _hContextMenu:       (e: Event)         => void  = () => {};
    private _hPointerLockChange: ()                 => void  = () => {};

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    init(canvas: HTMLCanvasElement): void {
        this._canvas = canvas;
        this._attachListeners(canvas);
    }

    /**
     * Must be called once at the start of each frame (before reading state).
     * Resets accumulated mouse deltas 
     */
    beginFrame(): void {
        // Snapshot the inter-frame accumulator into the readable fields,
        // then reset the accumulator for the next frame's events.
        this._mouse.dx = this._rawMouseDx;
        this._mouse.dy = this._rawMouseDy;
        this._mouse.scrollDelta = this._rawScrollDelta;
        this._rawMouseDx = 0;
        this._rawMouseDy = 0;
        this._rawScrollDelta = 0;

        // Poll gamepad (no events — the Gamepad API requires polling).
        this._pollGamepad();
    }

    private _pollGamepad(): void {
        const gamepads = navigator.getGamepads();
        let found = false;
        for (let i = 0; i < gamepads.length; i++) {
            const gp = gamepads[i];
            if (!gp || !gp.connected) continue;
            found = true;
            // Read axes with deadzone
            const axes = this._gamepad.axes;
            for (let a = 0; a < Math.min(gp.axes.length, 4); a++) {
                const raw = gp.axes[a]!;
                axes[a] = Math.abs(raw) < GAMEPAD_DEADZONE ? 0 : raw;
            }
            // Read buttons
            const btns = this._gamepad.buttons;
            btns.length = gp.buttons.length;
            for (let b = 0; b < gp.buttons.length; b++) {
                btns[b] = gp.buttons[b]!.pressed;
            }
            break; // use first connected gamepad
        }
        this._gamepad.connected = found;
    }

    /** Returns the current input state.  The object is reused — do not cache it across frames. */
    getState(): InputState {
        return {
            keyboard: this._keyboard,
            mouse:    this._mouse,
            gamepad:  this._gamepad,
        };
    }

    /**
     * Disable all input processing.
     * Event listeners remain attached but no state is updated.
     * Any held keys / mouse buttons are cleared immediately.
     */
    setEnabled(enabled: boolean): void {
        this._enabled = enabled;
        if (!enabled) {
            this._keyboard.keys.clear();
            this._mouse.buttons = 0;
            this._mouse.dx      = 0;
            this._mouse.dy      = 0;
            this._mouse.scrollDelta = 0;
            this._rawMouseDx    = 0;
            this._rawMouseDy    = 0;
            this._rawScrollDelta = 0;
        }
    }

    isEnabled(): boolean { return this._enabled; }

    destroy(): void {
        this._detachListeners();
        this._canvas = null;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DOM event wiring
    // ─────────────────────────────────────────────────────────────────────────

    private _attachListeners(canvas: HTMLCanvasElement): void {

        this._hKeyDown = (e: KeyboardEvent) => {
            if (!this._enabled) return;
            this._keyboard.keys.add(e.code);
        };

        this._hKeyUp = (e: KeyboardEvent) => {
            // Always remove on keyup (even when disabled) to avoid stuck keys.
            this._keyboard.keys.delete(e.code);
        };

        // mousemove on document captures movement during pointer lock and
        // also outside the canvas boundary during drag operations.
        this._hMouseMove = (e: MouseEvent) => {
            if (!this._enabled) return;
            this._mouse.x = e.clientX;
            this._mouse.y = e.clientY;
            // Accumulate movement deltas when pointer is locked, when the
            // event originates from the canvas, OR when the user is dragging
            // (button pressed on canvas, not yet released).
            if (this._mouse.locked || this._dragging || e.target === canvas) {
                this._rawMouseDx += e.movementX;
                this._rawMouseDy += e.movementY;
            }
        };

        this._hMouseDown = (e: MouseEvent) => {
            if (!this._enabled) return;
            this._mouse.buttons |= (1 << e.button);
            this._dragging = true;
            if (this.pointerLockEnabled) {
                canvas.requestPointerLock();
            }
        };

        this._hWheel = (e: WheelEvent) => {
            if (!this._enabled) return;
            e.preventDefault();
            this._rawScrollDelta += e.deltaY;
        };

        this._hMouseUp = (e: MouseEvent) => {
            this._mouse.buttons &= ~(1 << e.button);
            // Clear drag state when all buttons are released.
            if (this._mouse.buttons === 0) {
                this._dragging = false;
            }
        };

        this._hContextMenu = (e: Event) => e.preventDefault();

        this._hPointerLockChange = () => {
            this._mouse.locked = document.pointerLockElement === canvas;
        };

        window.addEventListener('keydown',   this._hKeyDown);
        window.addEventListener('keyup',     this._hKeyUp);
        document.addEventListener('mousemove', this._hMouseMove);
        canvas.addEventListener('mousedown', this._hMouseDown);
        canvas.addEventListener('wheel', this._hWheel, { passive: false });
        window.addEventListener('mouseup',   this._hMouseUp);
        canvas.addEventListener('contextmenu', this._hContextMenu);
        document.addEventListener('pointerlockchange', this._hPointerLockChange);
    }

    private _detachListeners(): void {
        const c = this._canvas;
        window.removeEventListener('keydown', this._hKeyDown);
        window.removeEventListener('keyup',   this._hKeyUp);
        document.removeEventListener('mousemove', this._hMouseMove);
        if (c) {
            c.removeEventListener('mousedown',    this._hMouseDown);
            c.removeEventListener('wheel',        this._hWheel);
            c.removeEventListener('contextmenu',  this._hContextMenu);
        }
        window.removeEventListener('mouseup', this._hMouseUp);
        document.removeEventListener('pointerlockchange', this._hPointerLockChange);
    }
}
