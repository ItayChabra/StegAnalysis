import { useRef, useState } from 'react';
import styles from './DemoBar.module.css';
import { buttonsFor, randomFrom } from './samples.js';

// Sample loader with separate buttons per pool (e.g. "Clean photo" / "Hidden
// data"). Each click fetches a *different* random image from that pool and hands
// it to the active flow as a File via onPick.
export default function DemoBar({ onPick, kind = 'cover' }) {
  const [loading, setLoading] = useState(null); // button key currently loading
  const lastRef = useRef({});                    // pool key -> last file shown

  const buttons = buttonsFor(kind);
  if (buttons.length === 0) return null;

  async function load(btn) {
    const file = randomFrom(btn.pool, lastRef.current[btn.key]);
    if (!file) return;
    lastRef.current[btn.key] = file;
    setLoading(btn.key);
    try {
      const res = await fetch(`/demo/${file}?t=${Date.now()}`);
      if (!res.ok) throw new Error('missing');
      const blob = await res.blob();
      onPick(new File([blob], file, { type: blob.type || 'image/png' }));
    } catch {
      // sample asset not present — ignore
    } finally {
      setLoading(null);
    }
  }

  return (
    <div className={styles.bar}>
      <span className={styles.label}>Try a sample:</span>
      {buttons.map((btn) => (
        <button
          key={btn.key}
          className={styles.chip}
          onClick={() => load(btn)}
          disabled={loading === btn.key}
        >
          {loading === btn.key ? '…' : btn.label}
        </button>
      ))}
      <span className={styles.hint}>Click again for a new one</span>
    </div>
  );
}
