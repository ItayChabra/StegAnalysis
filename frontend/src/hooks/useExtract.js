import { useCallback, useState } from 'react';
import { extractMessage } from '../api/client.js';

// State machine: IDLE → PROCESSING → COMPLETE | ERROR
// On ERROR, error.code is 'bad_key' | 'no_payload' | null for clear messaging.
export default function useExtract() {
  const [state, setState]   = useState('IDLE');
  const [result, setResult] = useState(null);
  const [error, setError]   = useState(null);

  const extract = useCallback(async (file, opts) => {
    setState('PROCESSING');
    setResult(null);
    setError(null);
    try {
      const data = await extractMessage(file, opts);
      setResult(data);
      setState('COMPLETE');
      return data;
    } catch (err) {
      setError(err);
      setState('ERROR');
      throw err;
    }
  }, []);

  const reset = useCallback(() => {
    setState('IDLE');
    setResult(null);
    setError(null);
  }, []);

  return { state, result, error, extract, reset };
}
