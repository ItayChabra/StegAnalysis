import { describe, it, expect } from 'vitest';
import { METHODS, METHOD_ORDER, defaultParams, formatHint } from './methods.js';

describe('METHODS / METHOD_ORDER', () => {
  it('every id in METHOD_ORDER has a matching METHODS entry', () => {
    for (const id of METHOD_ORDER) {
      expect(METHODS[id]).toBeDefined();
      expect(METHODS[id].id).toBe(id);
    }
  });

  it('has no duplicate entries', () => {
    expect(new Set(METHOD_ORDER).size).toBe(METHOD_ORDER.length);
  });

  it('every method has a plain-English label distinct from its technical name', () => {
    // Domain glossary rule (CLAUDE.md §3): UI must never surface raw ML terms.
    for (const id of METHOD_ORDER) {
      const m = METHODS[id];
      expect(m.plain).toBeTruthy();
      expect(m.plain).not.toBe(m.name);
    }
  });

  it('exactly lsb/dct/fft are recoverable; adaptive/steganogan are not', () => {
    const recoverable = METHOD_ORDER.filter((id) => METHODS[id].recoverable);
    expect(recoverable.sort()).toEqual(['dct', 'fft', 'lsb']);
  });

  it('every param has a key, label, type, and default', () => {
    for (const id of METHOD_ORDER) {
      for (const p of METHODS[id].params) {
        expect(p.key).toBeTruthy();
        expect(p.label).toBeTruthy();
        expect(['select', 'number']).toContain(p.type);
        expect(p.default).not.toBeUndefined();
      }
    }
  });

  it('every select param lists its default among its options', () => {
    for (const id of METHOD_ORDER) {
      for (const p of METHODS[id].params) {
        if (p.type !== 'select') continue;
        const values = p.options.map((o) => o.value);
        expect(values).toContain(p.default);
      }
    }
  });
});

describe('defaultParams', () => {
  it('builds a {key: default} object for a method with params', () => {
    expect(defaultParams('lsb')).toEqual({
      strategy: 'sequential',
      step: 1,
      bit_depth: 1,
    });
  });

  it('returns an empty object for a method with no configurable params', () => {
    expect(defaultParams('steganogan')).toEqual({});
  });

  it('covers every method in METHOD_ORDER without throwing', () => {
    for (const id of METHOD_ORDER) {
      expect(() => defaultParams(id)).not.toThrow();
    }
  });
});

describe('formatHint', () => {
  it('formats a recoverable method + cipher hint', () => {
    expect(formatHint({ method: 'lsb', cipher: 'aes256gcm' })).toBe('Pixel-level hiding · aes256gcm');
  });

  it('renders "no encryption" for cipher none', () => {
    expect(formatHint({ method: 'dct', cipher: 'none' })).toBe('JPEG frequency hiding · no encryption');
  });

  it('renders "no encryption" when cipher is missing entirely', () => {
    expect(formatHint({ method: 'fft' })).toBe('Frequency-domain hiding · no encryption');
  });

  it('falls back to the raw method id for an unknown method', () => {
    expect(formatHint({ method: 'not-a-real-method', cipher: 'none' })).toBe(
      'not-a-real-method · no encryption'
    );
  });

  it('returns an empty string for a null/undefined hint', () => {
    expect(formatHint(null)).toBe('');
    expect(formatHint(undefined)).toBe('');
  });
});
