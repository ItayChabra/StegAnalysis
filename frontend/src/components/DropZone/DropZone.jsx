import { useState, useRef, useEffect, useCallback } from 'react';
import styles from './DropZone.module.css';
import ui from '../../styles/ui.module.css';
import { pgmFileToDataUrl } from '../../utils/pgm.js';

const MAX_SIZE    = 20 * 1024 * 1024; // 20 MB
const ACCEPT_MIME = ['image/png', 'image/jpeg', 'image/bmp'];

function validate(file) {
  const name = file.name.toLowerCase();
  const isPgm = name.endsWith('.pgm');
  const isBmp = name.endsWith('.bmp') || file.type === 'image/bmp';
  const isMime = ACCEPT_MIME.includes(file.type);
  if (!isPgm && !isBmp && !isMime) return 'Unsupported file type. Please drop a PNG, JPG, BMP, or PGM image.';
  if (file.size > MAX_SIZE) return 'File exceeds the 20 MB limit.';
  return null;
}

export default function DropZone({ analyze, state }) {
  const [file,       setFile]       = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isPgm,      setIsPgm]      = useState(false);
  const [dragOver,   setDragOver]   = useState(false);
  const [validErr,   setValidErr]   = useState(null);
  const inputRef = useRef(null);

  const isDisabled = state !== 'IDLE';
  const isBusy     = state === 'UPLOADING' || state === 'ANALYZING';

  // Build preview whenever file changes. PGMs are decoded client-side.
  useEffect(() => {
    if (!file) { setPreviewUrl(null); return; }
    if (file.name.toLowerCase().endsWith('.pgm')) {
      let cancelled = false;
      pgmFileToDataUrl(file).then((url) => { if (!cancelled) setPreviewUrl(url); });
      return () => { cancelled = true; };
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const acceptFile = useCallback((incoming) => {
    const err = validate(incoming);
    if (err) { setValidErr(err); return; }
    setValidErr(null);
    setIsPgm(incoming.name.toLowerCase().endsWith('.pgm'));
    setFile(incoming);
  }, []);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    if (!isDisabled) setDragOver(true);
  }, [isDisabled]);

  const onDragLeave = useCallback(() => setDragOver(false), []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    if (isDisabled) return;
    const dropped = e.dataTransfer.files[0];
    if (dropped) acceptFile(dropped);
  }, [isDisabled, acceptFile]);

  const onZoneClick = useCallback(() => {
    if (isDisabled || file) return;
    inputRef.current?.click();
  }, [isDisabled, file]);

  const onInputChange = useCallback((e) => {
    const picked = e.target.files[0];
    if (picked) acceptFile(picked);
    e.target.value = '';
  }, [acceptFile]);

  const onAnalyze = useCallback(() => {
    if (file) analyze(file);
  }, [file, analyze]);

  const zoneClass = [
    styles.zone,
    dragOver   ? styles.dragOver : '',
    file       ? styles.hasFile  : '',
    isDisabled ? styles.disabled : '',
  ].filter(Boolean).join(' ');

  return (
    <div>
      <div
        className={zoneClass}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={onZoneClick}
        role={file ? undefined : 'button'}
        tabIndex={file || isDisabled ? -1 : 0}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onZoneClick(); }}
        aria-label="Drop image here or click to browse"
      >
        <input
          ref={inputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.bmp,.pgm"
          className={styles.fileInput}
          onChange={onInputChange}
          tabIndex={-1}
        />

        {file ? (
          /* ── File selected ── */
          <>
            {previewUrl ? (
              <img src={previewUrl} alt="Preview" className={styles.preview} />
            ) : isPgm ? (
              <div className={styles.pgmPlaceholder} aria-label="PGM image loaded">
                <div className={styles.pgmIcon} aria-hidden="true">
                  <div className={styles.pgmBar} />
                  <div className={styles.pgmBar} />
                  <div className={styles.pgmBar} />
                </div>
                <p className={styles.pgmLabel}>PGM Image Loaded</p>
              </div>
            ) : null}
            <p className={styles.filename}>{file.name}</p>
            <button
              className={ui.btnPrimary}
              style={{ width: 'auto' }}
              onClick={(e) => { e.stopPropagation(); onAnalyze(); }}
              disabled={isBusy}
            >
              {isBusy ? (
                <>
                  <span className={ui.spinner} aria-hidden="true" />
                  Analyzing…
                </>
              ) : (
                'Analyze →'
              )}
            </button>
          </>
        ) : (
          /* ── Empty state ── */
          <>
            <div className={styles.uploadIcon} aria-hidden="true">
              <div className={styles.arrowHead} />
              <div className={styles.arrowShaft} />
              <div className={styles.uploadTray} />
            </div>
            <p className={styles.hint}>Drop an image here</p>
            <p className={styles.formats}>PNG · JPG · BMP · PGM · max 20 MB</p>
          </>
        )}
      </div>

      {validErr && (
        <p className={styles.errorMsg} role="alert">{validErr}</p>
      )}
    </div>
  );
}
