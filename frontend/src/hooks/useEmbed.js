import { useCallback, useState } from 'react';
import { embedMessage } from '../api/client.js';

// State machine: IDLE → PROCESSING → COMPLETE | ERROR
export default function useEmbed() {
  const [state, setState]   = useState('IDLE');
  const [result, setResult] = useState(null);
  const [error, setError]   = useState(null);

  const embed = useCallback(async (file, opts) => {
    setState('PROCESSING');
    setResult(null);
    setError(null);
    try {
      const data = await embedMessage(file, opts);
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

  return { state, result, error, embed, reset };
}
