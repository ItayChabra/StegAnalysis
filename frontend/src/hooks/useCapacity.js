import { useCallback, useRef, useState } from 'react';
import { estimateCapacity } from '../api/client.js';

// Debounced capacity estimator for the embed form. Calls /api/capacity and
// exposes the latest { capacity_bytes, max_message_bytes, ... } for the meter.
export default function useCapacity(delay = 350) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const timer = useRef(null);
  const seq = useRef(0);

  const estimate = useCallback((file, method, params) => {
    if (timer.current) clearTimeout(timer.current);
    if (!file) { setData(null); return; }
    setLoading(true);
    const mySeq = ++seq.current;
    timer.current = setTimeout(async () => {
      try {
        const res = await estimateCapacity(file, method, params);
        if (mySeq === seq.current) setData(res);
      } catch {
        if (mySeq === seq.current) setData(null);
      } finally {
        if (mySeq === seq.current) setLoading(false);
      }
    }, delay);
  }, [delay]);

  const clear = useCallback(() => { setData(null); setLoading(false); }, []);

  return { data, loading, estimate, clear };
}
