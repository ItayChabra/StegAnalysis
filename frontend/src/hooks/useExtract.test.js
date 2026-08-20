import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import useExtract from './useExtract.js';
import { extractMessage, decryptMessage } from '../api/client.js';

vi.mock('../api/client.js', () => ({
  extractMessage: vi.fn(),
  decryptMessage: vi.fn(),
}));

beforeEach(() => {
  vi.clearAllMocks();
});

const REVEAL_RESULT = {
  method: 'lsb', cipher: 'aes256gcm', encrypted: true, bytes: 42,
  ciphertext_b64: 'Y2lwaGVy', salt_b64: 'c2FsdA==', nonce_b64: 'bm9uY2U=',
};

describe('useExtract — extract step', () => {
  it('starts IDLE', () => {
    const { result } = renderHook(() => useExtract());
    expect(result.current.state).toBe('IDLE');
  });

  it('PROCESSING -> REVEALED on a found payload', async () => {
    extractMessage.mockResolvedValueOnce(REVEAL_RESULT);
    const { result } = renderHook(() => useExtract());

    let p;
    act(() => { p = result.current.extract(new File([], 's.png'), { method: 'lsb' }); });
    expect(result.current.state).toBe('PROCESSING');
    await act(async () => { await p; });

    expect(result.current.state).toBe('REVEALED');
    expect(result.current.result).toEqual(REVEAL_RESULT);
  });

  it('PROCESSING -> ERROR with code "no_payload" when nothing is found', async () => {
    const err = new Error('[404] No recoverable message found.');
    err.status = 404;
    err.code = 'no_payload';
    extractMessage.mockRejectedValueOnce(err);
    const { result } = renderHook(() => useExtract());

    await act(async () => {
      try { await result.current.extract(new File([], 's.png'), {}); } catch { /* expected */ }
    });

    expect(result.current.state).toBe('ERROR');
    expect(result.current.error.code).toBe('no_payload');
  });
});

describe('useExtract — decrypt step', () => {
  async function revealedHook() {
    extractMessage.mockResolvedValueOnce(REVEAL_RESULT);
    const rendered = renderHook(() => useExtract());
    await act(async () => { await rendered.result.current.extract(new File([], 's.png'), {}); });
    return rendered;
  }

  it('throws synchronously if called before a successful extract', async () => {
    const { result } = renderHook(() => useExtract());
    await expect(result.current.decrypt('pw')).rejects.toThrow('Nothing to decrypt yet.');
  });

  it('REVEALED -> DECRYPTING -> DECRYPTED on the right passphrase', async () => {
    const { result } = await revealedHook();
    decryptMessage.mockResolvedValueOnce({ message: 'the secret', cipher: 'aes256gcm', bytes: 10 });

    let p;
    act(() => { p = result.current.decrypt('right-pw'); });
    expect(result.current.state).toBe('DECRYPTING');
    await act(async () => { await p; });

    expect(result.current.state).toBe('DECRYPTED');
    expect(result.current.plaintext.message).toBe('the secret');
    expect(decryptMessage).toHaveBeenCalledWith({
      ciphertext_b64: REVEAL_RESULT.ciphertext_b64,
      cipher: REVEAL_RESULT.cipher,
      passphrase: 'right-pw',
      salt_b64: REVEAL_RESULT.salt_b64,
      nonce_b64: REVEAL_RESULT.nonce_b64,
    });
  });

  it('a wrong passphrase drops back to REVEALED (not ERROR) so the user can retry', async () => {
    const { result } = await revealedHook();
    const err = new Error('[422] Wrong passphrase or the data has been altered.');
    err.code = 'bad_key';
    decryptMessage.mockRejectedValueOnce(err);

    await act(async () => {
      try { await result.current.decrypt('wrong-pw'); } catch { /* expected */ }
    });

    expect(result.current.state).toBe('REVEALED');
    expect(result.current.error.code).toBe('bad_key');
    expect(result.current.plaintext).toBeNull();
  });

  it('defaults to an empty-string passphrase when none is given', async () => {
    const { result } = await revealedHook();
    decryptMessage.mockResolvedValueOnce({ message: 'x' });
    await act(async () => { await result.current.decrypt(undefined); });
    expect(decryptMessage).toHaveBeenCalledWith(expect.objectContaining({ passphrase: '' }));
  });
});

describe('useExtract — reset', () => {
  it('clears state/result/plaintext/error back to IDLE', async () => {
    extractMessage.mockResolvedValueOnce(REVEAL_RESULT);
    const { result } = renderHook(() => useExtract());
    await act(async () => { await result.current.extract(new File([], 's.png'), {}); });
    expect(result.current.state).toBe('REVEALED');

    act(() => { result.current.reset(); });

    expect(result.current.state).toBe('IDLE');
    expect(result.current.result).toBeNull();
    expect(result.current.plaintext).toBeNull();
    expect(result.current.error).toBeNull();
  });
});
