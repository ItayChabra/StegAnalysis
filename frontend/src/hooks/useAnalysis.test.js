import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import useAnalysis from './useAnalysis.js';
import { analyzeImage } from '../api/client.js';

vi.mock('../api/client.js', () => ({
  analyzeImage: vi.fn(),
}));

beforeEach(() => {
  vi.clearAllMocks();
});

describe('useAnalysis', () => {
  it('starts IDLE', () => {
    const { result } = renderHook(() => useAnalysis());
    expect(result.current.state).toBe('IDLE');
  });

  it('passes through UPLOADING -> ANALYZING -> COMPLETE on success', async () => {
    const fakeResponse = { job_id: 'j1', verdict: 'CLEAN', confidence: 0.12 };
    analyzeImage.mockResolvedValueOnce(fakeResponse);
    const { result } = renderHook(() => useAnalysis());

    act(() => { result.current.analyze(new File([], 'x.png')); });
    expect(result.current.state).toBe('UPLOADING');

    // There's a deliberate ~600ms ANALYZING beat before COMPLETE (see the hook).
    await waitFor(() => expect(result.current.state).toBe('ANALYZING'));
    await waitFor(() => expect(result.current.state).toBe('COMPLETE'), { timeout: 2000 });

    expect(result.current.result).toEqual(fakeResponse);
    expect(result.current.error).toBeNull();
  });

  it('goes straight to ERROR on failure, without the ANALYZING beat', async () => {
    const err = new Error('[400] Invalid image file');
    analyzeImage.mockRejectedValueOnce(err);
    const { result } = renderHook(() => useAnalysis());

    await act(async () => { await result.current.analyze(new File([], 'x.png')); });

    expect(result.current.state).toBe('ERROR');
    expect(result.current.error).toBe(err);
    expect(result.current.result).toBeNull();
  });

  it('reset() returns to IDLE', async () => {
    analyzeImage.mockRejectedValueOnce(new Error('fails'));
    const { result } = renderHook(() => useAnalysis());
    await act(async () => { await result.current.analyze(new File([], 'x.png')); });
    expect(result.current.state).toBe('ERROR');

    act(() => { result.current.reset(); });
    expect(result.current.state).toBe('IDLE');
    expect(result.current.error).toBeNull();
  });
});
