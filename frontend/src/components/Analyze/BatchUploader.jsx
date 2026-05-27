import { useEffect, useRef, useState } from 'react';
import ui from '../../styles/ui.module.css';
import styles from './BatchUploader.module.css';
import { pgmFileToDataUrl } from '../../utils/pgm.js';

const MAX_SIZE = 20 * 1024 * 1024;

function ok(f) {
  const name = f.name.toLowerCase();
  return name.endsWith('.pgm')
    || name.endsWith('.bmp')
    || ['image/png', 'image/jpeg', 'image/bmp'].includes(f.type);
}

// Multi-file picker for batch analysis. Shows a thumbnail of every selected
// image so it's obvious they were added, then runs them.
export default function BatchUploader({ onRun, running }) {
  const [files, setFiles] = useState([]);
  const [dragOver, setDragOver] = useState(false);
  const [previews, setPreviews] = useState({}); // key -> URL string or null
  const inputRef = useRef(null);

  const keyOf = (f) => `${f.name}:${f.size}`;

  // Build previews as the file list changes. PGMs are parsed asynchronously.
  useEffect(() => {
    let cancelled = false;

    async function build() {
      const next = {};
      await Promise.all(files.map(async (f) => {
        const k = keyOf(f);
        if (previews[k] !== undefined) { next[k] = previews[k]; return; }
        if (f.name.toLowerCase().endsWith('.pgm')) {
          next[k] = await pgmFileToDataUrl(f); // data-URL or null on failure
        } else {
          next[k] = URL.createObjectURL(f);
        }
      }));
      if (cancelled) return;
      // Revoke blob URLs that are no longer in the list
      Object.entries(previews).forEach(([k, url]) => {
        if (url && url.startsWith('blob:') && next[k] === undefined) URL.revokeObjectURL(url);
      });
      setPreviews(next);
    }

    build();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files]);

  // Revoke all blob URLs on unmount
  useEffect(() => () => {
    Object.values(previews).forEach((url) => {
      if (url && url.startsWith('blob:')) URL.revokeObjectURL(url);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function add(list) {
    const valid = Array.from(list).filter((f) => ok(f) && f.size <= MAX_SIZE);
    setFiles((prev) => {
      const seen = new Set(prev.map(keyOf));
      return [...prev, ...valid.filter((f) => !seen.has(keyOf(f)))];
    });
  }

  function removeAt(idx) {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  }

  return (
    <div className={ui.card}>
      <div
        className={`${styles.zone}${dragOver ? ` ${styles.dragOver}` : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); add(e.dataTransfer.files); }}
        onClick={() => inputRef.current?.click()}
        role="button" tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click(); }}
      >
        <input
          ref={inputRef} type="file" accept=".png,.jpg,.jpeg,.bmp,.pgm" multiple
          className={styles.input}
          onChange={(e) => { add(e.target.files); e.target.value = ''; }}
        />
        <div className={styles.icon}>↑</div>
        <p className={styles.label}>Drop one or more images</p>
        <p className={styles.formats}>PNG · JPG · BMP · PGM · select multiple to batch</p>
      </div>

      {files.length > 0 && (
        <>
          <div className={styles.thumbs}>
            {files.map((f, i) => (
              <div key={keyOf(f)} className={styles.thumb}>
                {previews[keyOf(f)]
                  ? <img src={previews[keyOf(f)]} alt={f.name} className={styles.thumbImg} />
                  : <div className={styles.thumbPgm}>PGM</div>}
                <button
                  className={styles.remove}
                  onClick={(e) => { e.stopPropagation(); removeAt(i); }}
                  disabled={running}
                  aria-label={`Remove ${f.name}`}
                  title="Remove"
                >×</button>
                <span className={styles.thumbName}>{f.name}</span>
              </div>
            ))}
          </div>

          <div className={styles.selected}>
            <span className={styles.count}>{files.length} image{files.length > 1 ? 's' : ''} ready</span>
            <div className={styles.actions}>
              <button className={ui.btn} onClick={() => setFiles([])} disabled={running}>Clear</button>
              <button className={ui.btnPrimary} style={{ width: 'auto' }} onClick={() => onRun(files)} disabled={running}>
                {running ? <><span className={ui.spinner} /> Analyzing…</> : `Analyze ${files.length} →`}
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
