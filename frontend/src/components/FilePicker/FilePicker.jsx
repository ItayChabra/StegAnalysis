import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './FilePicker.module.css';
import { pgmFileToDataUrl } from '../../utils/pgm.js';

const MAX_SIZE = 20 * 1024 * 1024;
const ACCEPT_MIME = ['image/png', 'image/jpeg', 'image/bmp'];

function validate(file) {
  const name = file.name.toLowerCase();
  const isPgm = name.endsWith('.pgm');
  const isBmp = name.endsWith('.bmp') || file.type === 'image/bmp';
  if (!isPgm && !isBmp && !ACCEPT_MIME.includes(file.type))
    return 'Unsupported file. Use PNG, JPG, BMP, or PGM.';
  if (file.size > MAX_SIZE) return 'File exceeds the 20 MB limit.';
  return null;
}

// Generic single-file drag/drop with preview. Parent owns the file via onFile.
export default function FilePicker({ file, onFile, label = 'Drop an image', hint, compact }) {
  const [dragOver, setDragOver] = useState(false);
  const [err, setErr] = useState(null);
  const [preview, setPreview] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!file) { setPreview(null); return; }
    if (file.name.toLowerCase().endsWith('.pgm')) {
      let cancelled = false;
      pgmFileToDataUrl(file).then((url) => { if (!cancelled) setPreview(url); });
      return () => { cancelled = true; };
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const accept = useCallback((incoming) => {
    const v = validate(incoming);
    if (v) { setErr(v); return; }
    setErr(null);
    onFile(incoming);
  }, [onFile]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) accept(f);
  }, [accept]);

  return (
    <div>
      <div
        className={[
          styles.zone,
          compact ? styles.compact : '',
          dragOver ? styles.dragOver : '',
          file ? styles.hasFile : '',
        ].filter(Boolean).join(' ')}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click(); }}
        aria-label={label}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.bmp,.pgm"
          className={styles.input}
          onChange={(e) => { if (e.target.files[0]) accept(e.target.files[0]); e.target.value = ''; }}
          tabIndex={-1}
        />
        {file ? (
          <>
            {preview
              ? <img src={preview} alt="Preview" className={styles.preview} />
              : <div className={styles.pgm}>PGM image loaded</div>}
            <p className={styles.filename}>{file.name}</p>
          </>
        ) : (
          <>
            <div className={styles.icon} aria-hidden="true">↑</div>
            <p className={styles.label}>{label}</p>
            <p className={styles.formats}>PNG · JPG · BMP · PGM · max 20 MB</p>
          </>
        )}
      </div>
      {hint && !err && <p className={styles.hint}>{hint}</p>}
      {err && <p className={styles.err} role="alert">{err}</p>}
    </div>
  );
}
