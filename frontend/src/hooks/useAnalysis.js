import { useState, useCallback } from 'react';
import { analyzeImage } from '../api/client.js';

// State machine: IDLE → UPLOADING → ANALYZING → COMPLETE
//                Any state → ERROR

export default function useAnalysis() {
  const [state,  setState]  = useState('IDLE');
  const [result, setResult] = useState(null);
  const [error,  setError]  = useState(null);

  const analyze = useCallback(async (file) => {
    setState('UPLOADING');
    setResult(null);
    setError(null);

    try {
      const data = await analyzeImage(file);

      // Brief ANALYZING beat so the transition to COMPLETE feels intentional.
      setState('ANALYZING');
      await new Promise((resolve) => setTimeout(resolve, 600));

      setResult(data);
      setState('COMPLETE');
    } catch (err) {
      setError(err);
      setState('ERROR');
    }
  }, []);

  const reset = useCallback(() => {
    setState('IDLE');
    setResult(null);
    setError(null);
  }, []);

  return { state, result, error, analyze, reset };
}
