// /src/engine/core/Logger.ts

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const PRIORITY: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 };

// Colour used for the tag label in info/debug messages
const TAG_STYLE: Record<LogLevel, string> = {
    debug: 'color:#888;font-weight:bold',
    info:  'color:#4af;font-weight:bold',
    warn:  'color:#fa4;font-weight:bold',
    error: 'color:#f44;font-weight:bold',
};

let _globalLevel: LogLevel = 'debug';

/**
 * Lightweight named logger.
 *
 * Each engine subsystem creates its own instance:
 *
 *   private readonly _log = new Logger('GPUBackend');
 *
 * Global minimum level can be changed at any time:
 *
 *   Logger.setLevel('warn');   // silence debug + info
 */
export class Logger {

    private readonly _tag: string;

    constructor(tag: string) {
        this._tag = tag;
    }

    // -------------------------------------------------------------------------
    // Global level control
    // -------------------------------------------------------------------------

    static setLevel(level: LogLevel): void {
        _globalLevel = level;
    }

    static getLevel(): LogLevel {
        return _globalLevel;
    }

    // -------------------------------------------------------------------------
    // Log methods
    // -------------------------------------------------------------------------

    debug(...args: unknown[]): void {
        if (PRIORITY['debug'] < PRIORITY[_globalLevel]) return;
        console.debug(`%c[${this._tag}]`, TAG_STYLE.debug, ...args);
    }

    info(...args: unknown[]): void {
        if (PRIORITY['info'] < PRIORITY[_globalLevel]) return;
        console.info(`%c[${this._tag}]`, TAG_STYLE.info, ...args);
    }

    warn(...args: unknown[]): void {
        if (PRIORITY['warn'] < PRIORITY[_globalLevel]) return;
        console.warn(`%c[${this._tag}]`, TAG_STYLE.warn, ...args);
    }

    /** Errors are always emitted regardless of global level. */
    error(...args: unknown[]): void {
        console.error(`%c[${this._tag}]`, TAG_STYLE.error, ...args);
    }
}
