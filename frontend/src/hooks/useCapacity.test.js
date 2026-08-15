import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import useCapacity from './useCapacity.js';
import { estimateCapacity } from '../api/client.js';

vi.mock('../api/client.js', () => ({
  estimateCapacity: vi.fn(),
}));

beforeEach(() => {
  vi.clearAllMocks();
});

describe('useCapacity', () => {
  it('starts with no data, not loading', () => {
    const { result } = renderHook(() => useCapacity(0));
    expect(result.current.data).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  it('debounces: does not call the API before the delay elapses', async () => {
    estimateCapacity.mockResolvedValue({ capacity_bytes: 100 });
    const { result } = renderHook(() => useCapacity(50));

    act(() => { result.current.estimate(new File([], 'c.png'), 'lsb', {}); });
    expect(result.current.loading).toBe(true);
    expect(estimateCapacity).not.toHaveBeenCalled();

    await waitFor(() => expect(estimateCapacity).toHaveBeenCalledTimes(1));
  });

  it('sets data after the debounced call resolves', async () => {
    const fakeCapacity = { capacity_bytes: 512, max_message_bytes: 460 };
    estimateCapacity.mockResolvedValue(fakeCapacity);
    const { result } = renderHook(() => useCapacity(0));

    act(() => { result.current.estimate(new File([], 'c.png'), 'lsb', {}); });
    await waitFor(() => expect(result.current.data).toEqual(fakeCapacity));
    expect(result.current.loading).toBe(false);
  });

  it('passing a null file clears data immediately, with no API call', () => {
    const { result } = renderHook(() => useCapacity(0));
    act(() => { result.current.estimate(null, 'lsb', {}); });
    expect(result.current.data).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(estimateCapacity).not.toHaveBeenCalled();
  });

  it('a failed request clears data rather than leaving stale data, and stops loading', async () => {
    estimateCapacity.mockRejectedValue(new Error('network error'));
    const { result } = renderHook(() => useCapacity(0));

    act(() => { result.current.estimate(new File([], 'c.png'), 'lsb', {}); });
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data).toBeNull();
  });

  it('only the latest of several rapid calls applies its result (stale responses are ignored)', async () => {
    // First call resolves SLOWER than the second — without the hook's seq-ref
    // guard, the stale first response would overwrite the newer, correct one.
    let resolveFirst;
    const firstPromise = new Promise((resolve) => { resolveFirst = resolve; });
    estimateCapacity
      .mockImplementationOnce(() => firstPromise)
      .mockImplementationOnce(() => Promise.resolve({ capacity_bytes: 999 }));

    const { result } = renderHook(() => useCapacity(0));

    act(() => { result.current.estimate(new File([], 'a.png'), 'lsb', {}); });
    await waitFor(() => expect(estimateCapacity).toHaveBeenCalledTimes(1));

    act(() => { result.current.estimate(new File([], 'b.png'), 'dct', {}); });
    await waitFor(() => expect(result.current.data).toEqual({ capacity_bytes: 999 }));

    // Now let the stale first call resolve — it must NOT clobber the newer data.
    await act(async () => { resolveFirst({ capacity_bytes: 111 }); await firstPromise; });
    expect(result.current.data).toEqual({ capacity_bytes: 999 });
  });

  it('clear() resets data and loading', async () => {
    estimateCapacity.mockResolvedValue({ capacity_bytes: 100 });
    const { result } = renderHook(() => useCapacity(0));
    act(() => { result.current.estimate(new File([], 'c.png'), 'lsb', {}); });
    await waitFor(() => expect(result.current.data).not.toBeNull());

    act(() => { result.current.clear(); });

    expect(result.current.data).toBeNull();
    expect(result.current.loading).toBe(false);
  });
});
