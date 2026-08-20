import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import useEmbed from './useEmbed.js';
import { embedMessage } from '../api/client.js';

vi.mock('../api/client.js', () => ({
  embedMessage: vi.fn(),
}));

beforeEach(() => {
  vi.clearAllMocks();
});

describe('useEmbed', () => {
  it('starts IDLE with no result/error', () => {
    const { result } = renderHook(() => useEmbed());
    expect(result.current.state).toBe('IDLE');
    expect(result.current.result).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('goes PROCESSING -> COMPLETE and stores the result on success', async () => {
    const fakeResponse = { job_id: 'abc123', method: 'lsb', recoverable: true };
    embedMessage.mockResolvedValueOnce(fakeResponse);
    const { result } = renderHook(() => useEmbed());

    let embedPromise;
    act(() => {
      embedPromise = result.current.embed(new File([], 'cover.png'), { method: 'lsb' });
    });
    expect(result.current.state).toBe('PROCESSING');

    await act(async () => { await embedPromise; });

    expect(result.current.state).toBe('COMPLETE');
    expect(result.current.result).toEqual(fakeResponse);
    expect(result.current.error).toBeNull();
    expect(embedMessage).toHaveBeenCalledWith(expect.any(File), { method: 'lsb' });
  });

  it('goes PROCESSING -> ERROR and re-throws on failure', async () => {
    const err = new Error('[400] Message too large for this image and settings.');
    embedMessage.mockRejectedValueOnce(err);
    const { result } = renderHook(() => useEmbed());

    let caught = null;
    await act(async () => {
      try {
        await result.current.embed(new File([], 'cover.png'), {});
      } catch (e) {
        caught = e;
      }
    });

    expect(caught).toBe(err); // the hook must not swallow the error
    expect(result.current.state).toBe('ERROR');
    expect(result.current.error).toBe(err);
    expect(result.current.result).toBeNull();
  });

  it('reset() returns to IDLE and clears result/error', async () => {
    embedMessage.mockResolvedValueOnce({ job_id: 'x' });
    const { result } = renderHook(() => useEmbed());
    await act(async () => { await result.current.embed(new File([], 'c.png'), {}); });
    expect(result.current.state).toBe('COMPLETE');

    act(() => { result.current.reset(); });

    expect(result.current.state).toBe('IDLE');
    expect(result.current.result).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('a fresh embed() call clears a previous error', async () => {
    embedMessage.mockRejectedValueOnce(new Error('first call fails'));
    const { result } = renderHook(() => useEmbed());
    await act(async () => {
      try { await result.current.embed(new File([], 'c.png'), {}); } catch { /* expected */ }
    });
    expect(result.current.state).toBe('ERROR');

    embedMessage.mockResolvedValueOnce({ job_id: 'ok' });
    await act(async () => { await result.current.embed(new File([], 'c.png'), {}); });

    expect(result.current.state).toBe('COMPLETE');
    expect(result.current.error).toBeNull();
  });
});
