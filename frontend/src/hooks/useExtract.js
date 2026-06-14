import { useCallback, useState } from 'react';
import { extractMessage, decryptMessage } from '../api/client.js';

// Two-step reveal flow:
//   IDLE → PROCESSING → REVEALED | ERROR        (extract)
//   REVEALED → DECRYPTING → DECRYPTED | ERROR    (decrypt)
//
// `result` holds the reveal payload (ciphertext_b64, cipher, salt_b64, nonce_b64,
// encrypted, bytes, method, and `message` when not encrypted).
// `plaintext` holds the decrypted message once the user supplies a key.
// `error.code` is 'bad_key' | 'no_payload' | null for clear messaging.
export default function useExtract() {
  const [state, setState]         = useState('IDLE');
  const [result, setResult]       = useState(null);
  const [plaintext, setPlaintext] = useState(null);
  const [error, setError]         = useState(null);

  const extract = useCallback(async (file, opts) => {
    setState('PROCESSING');
    setResult(null);
    setPlaintext(null);
    setError(null);
    try {
      const data = await extractMessage(file, opts);
      setResult(data);
      setState('REVEALED');
      return data;
    } catch (err) {
      setError(err);
      setState('ERROR');
      throw err;
    }
  }, []);

  const decrypt = useCallback(async (passphrase) => {
    if (!result) throw new Error('Nothing to decrypt yet.');
    setState('DECRYPTING');
    setError(null);
    try {
      const data = await decryptMessage({
        ciphertext_b64: result.ciphertext_b64,
        cipher:         result.cipher,
        passphrase:     passphrase ?? '',
        salt_b64:       result.salt_b64 || '',
        nonce_b64:      result.nonce_b64 || '',
      });
      setPlaintext(data);
      setState('DECRYPTED');
      return data;
    } catch (err) {
      setError(err);
      setState('REVEALED');   // stay on the reveal screen so the user can retry
      throw err;
    }
  }, [result]);

  const reset = useCallback(() => {
    setState('IDLE');
    setResult(null);
    setPlaintext(null);
    setError(null);
  }, []);

  return { state, result, plaintext, error, extract, decrypt, reset };
}