import { useState, useRef, useEffect, useCallback } from 'react';
import styles from './ImageComparisonSlider.module.css';
import { originalUrl, noisemapUrl, heatmapUrl } from '../../api/client.js';

export default function ImageComparisonSlider({
  jobId,
  windowRows,
  windowCols,
  selectedPatch,
}) {
  const [dividerPct,   setDividerPct]   = useState(50);
  const [showNoise,    setShowNoise]     = useState(true);
  const [naturalW,     setNaturalW]      = useState(null);
  const [naturalH,     setNaturalH]      = useState(null);
  const [containerW,   setContainerW]    = useState(0);
  const [containerH,   setContainerH]    = useState(0);

  const isDragging  = useRef(false);
  const rafRef      = useRef(null);
  const containerRef = useRef(null);

  // Keep container dimensions in sync with resize
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setContainerW(width);
      setContainerH(height);
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // ── Drag helpers ─────────────────────────────────────────────────────────

  const computePct = useCallback((clientX) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const pct = ((clientX - rect.left) / rect.width) * 100;
    return Math.min(98, Math.max(2, pct));
  }, []);

  const onMouseDown = useCallback((e) => {
    e.preventDefault();
    isDragging.current = true;
  }, []);

  const onMouseMove = useCallback((e) => {
    if (!isDragging.current) return;
    const pct = computePct(e.clientX);
    if (pct == null) return;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => setDividerPct(pct));
  }, [computePct]);

  const onMouseUp = useCallback(() => { isDragging.current = false; }, []);

  const onTouchStart = useCallback((e) => {
    isDragging.current = true;
  }, []);

  const onTouchMove = useCallback((e) => {
    if (!isDragging.current) return;
    const pct = computePct(e.touches[0].clientX);
    if (pct == null) return;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => setDividerPct(pct));
  }, [computePct]);

  const onTouchEnd = useCallback(() => { isDragging.current = false; }, []);

  // ── Patch highlight geometry ──────────────────────────────────────────────

  let patchOverlay = null;
  if (
    selectedPatch != null &&
    naturalW && naturalH &&
    containerW && containerH &&
    windowRows && windowCols
  ) {
    const patchRow = Math.floor(selectedPatch / windowCols);
    const patchCol = selectedPatch % windowCols;

    const containerAspect = containerW / containerH;
    const imageAspect     = naturalW   / naturalH;
    let renderedW, renderedH, offsetX, offsetY;

    if (imageAspect > containerAspect) {
      renderedW = containerW;
      renderedH = containerW / imageAspect;
      offsetX   = 0;
      offsetY   = (containerH - renderedH) / 2;
    } else {
      renderedH = containerH;
      renderedW = containerH * imageAspect;
      offsetX   = (containerW - renderedW) / 2;
      offsetY   = 0;
    }

    const totalPatchW = windowCols * 256;
    const totalPatchH = windowRows * 256;
    const scaleX = renderedW / totalPatchW;
    const scaleY = renderedH / totalPatchH;

    patchOverlay = {
      left:   offsetX + patchCol * 256 * scaleX,
      top:    offsetY + patchRow * 256 * scaleY,
      width:  256 * scaleX,
      height: 256 * scaleY,
    };
  }

  const rightLabel = showNoise ? 'What the model sees' : 'Suspicion map';
  const rightSrc   = jobId
    ? (showNoise ? noisemapUrl(jobId) : heatmapUrl(jobId))
    : null;
  const leftSrc    = jobId ? originalUrl(jobId) : null;
  const rightClip  = `inset(0 ${100 - dividerPct}% 0 0)`;

  return (
    <div className={styles.wrapper}>
      {/* Toggle */}
      <div className={styles.toggleRow}>
        <button
          className={`${styles.toggleBtn}${showNoise ? ` ${styles.active}` : ''}`}
          onClick={() => setShowNoise(true)}
        >
          What the model sees
        </button>
        <button
          className={`${styles.toggleBtn}${!showNoise ? ` ${styles.active}` : ''}`}
          onClick={() => setShowNoise(false)}
        >
          Suspicion map
        </button>
      </div>

      {/* Panel */}
      <div
        ref={containerRef}
        className={styles.panel}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
      >
        {/* Left image — original */}
        {leftSrc && (
          <img
            src={leftSrc}
            alt="Original image"
            className={styles.imgLeft}
            onLoad={(e) => {
              setNaturalW(e.target.naturalWidth);
              setNaturalH(e.target.naturalHeight);
            }}
            draggable={false}
          />
        )}

        {/* Right image — noisemap or heatmap, clipped */}
        {rightSrc && (
          <img
            src={rightSrc}
            alt={rightLabel}
            className={styles.imgRight}
            style={{ clipPath: rightClip }}
            draggable={false}
          />
        )}

        {/* Patch highlight on left panel */}
        {patchOverlay && (
          <div
            className={styles.patchHighlight}
            style={{
              left:   patchOverlay.left,
              top:    patchOverlay.top,
              width:  patchOverlay.width,
              height: patchOverlay.height,
            }}
          />
        )}

        {/* Divider */}
        <div
          className={styles.dividerLine}
          style={{ left: `${dividerPct}%` }}
        />
        <div
          className={styles.handle}
          style={{ left: `${dividerPct}%` }}
          onMouseDown={onMouseDown}
          onTouchStart={onTouchStart}
          aria-label="Drag to compare images"
          role="slider"
          aria-valuenow={Math.round(dividerPct)}
          aria-valuemin={2}
          aria-valuemax={98}
        >
          ↔
        </div>

        {/* Labels */}
        <span className={styles.labelLeft}>What you see</span>
        <span className={styles.labelRight}>{rightLabel}</span>
      </div>
    </div>
  );
}
