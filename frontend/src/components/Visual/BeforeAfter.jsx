import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './BeforeAfter.module.css';

// Generic drag-to-compare slider built from raw pointer events + rAF.
// Optional patch overlay (256px-window grid) for the analyze suspicion view.
export default function BeforeAfter({
  leftSrc, rightSrc, leftLabel, rightLabel,
  windowRows, windowCols, selectedPatch,
}) {
  const [pct, setPct] = useState(50);
  const [natW, setNatW] = useState(null);
  const [natH, setNatH] = useState(null);
  const [cw, setCw] = useState(0);
  const [ch, setCh] = useState(0);
  const dragging = useRef(false);
  const raf = useRef(null);
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => {
      setCw(e.contentRect.width);
      setCh(e.contentRect.height);
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  const compute = useCallback((clientX) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return null;
    return Math.min(98, Math.max(2, ((clientX - rect.left) / rect.width) * 100));
  }, []);

  const move = useCallback((clientX) => {
    if (!dragging.current) return;
    const p = compute(clientX);
    if (p == null) return;
    if (raf.current) cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(() => setPct(p));
  }, [compute]);

  let overlay = null;
  if (selectedPatch != null && natW && natH && cw && ch && windowRows && windowCols) {
    const row = Math.floor(selectedPatch / windowCols);
    const col = selectedPatch % windowCols;
    const cAspect = cw / ch;
    const iAspect = natW / natH;
    let rW, rH, oX, oY;
    if (iAspect > cAspect) { rW = cw; rH = cw / iAspect; oX = 0; oY = (ch - rH) / 2; }
    else { rH = ch; rW = ch * iAspect; oX = (cw - rW) / 2; oY = 0; }
    const sX = rW / (windowCols * 256);
    const sY = rH / (windowRows * 256);
    overlay = { left: oX + col * 256 * sX, top: oY + row * 256 * sY, width: 256 * sX, height: 256 * sY };
  }

  return (
    <div
      ref={ref}
      className={styles.panel}
      onMouseMove={(e) => move(e.clientX)}
      onMouseUp={() => { dragging.current = false; }}
      onMouseLeave={() => { dragging.current = false; }}
      onTouchMove={(e) => move(e.touches[0].clientX)}
      onTouchEnd={() => { dragging.current = false; }}
    >
      {leftSrc && (
        <img
          src={leftSrc} alt={leftLabel} className={styles.imgLeft} draggable={false}
          onLoad={(e) => { setNatW(e.target.naturalWidth); setNatH(e.target.naturalHeight); }}
        />
      )}
      {rightSrc && (
        <img
          src={rightSrc} alt={rightLabel} className={styles.imgRight}
          style={{ clipPath: `inset(0 0 0 ${pct}%)` }} draggable={false}
        />
      )}
      {overlay && (
        <div className={styles.patch} style={overlay} />
      )}
      <div className={styles.divider} style={{ left: `${pct}%` }} />
      <div
        className={styles.handle}
        style={{ left: `${pct}%` }}
        onMouseDown={(e) => { e.preventDefault(); dragging.current = true; }}
        onTouchStart={() => { dragging.current = true; }}
        role="slider"
        aria-label="Drag to compare"
        aria-valuenow={Math.round(pct)} aria-valuemin={2} aria-valuemax={98}
      >↔</div>
      <span className={styles.labelLeft}>{leftLabel}</span>
      <span className={styles.labelRight}>{rightLabel}</span>
    </div>
  );
}
