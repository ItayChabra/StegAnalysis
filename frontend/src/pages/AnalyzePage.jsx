import { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import ui from '../styles/ui.module.css';
import styles from './AnalyzePage.module.css';
import BatchUploader from '../components/Analyze/BatchUploader.jsx';
import ImageResultCard from '../components/Analyze/ImageResultCard.jsx';
import Dashboard from '../components/Dashboard/Dashboard.jsx';
import VisualTools from '../components/Visual/VisualTools.jsx';
import BitPlaneExplorer from '../components/Visual/BitPlaneExplorer.jsx';
import DemoBar from '../components/Demo/DemoBar.jsx';
import useBatchAnalysis from '../hooks/useBatchAnalysis.js';
import { useApp } from '../context/AppContext.jsx';
import { downloadSanitized } from '../api/client.js';

const VERDICT_SUMMARY = {
  CLEAN: 'Clean', SUSPICIOUS: 'Suspicious', STEGO_DETECTED: 'Hidden data found',
};

export default function AnalyzePage() {
  const { items, running, run, restore } = useBatchAnalysis();
  const { history } = useApp();
  const location = useLocation();
  const navigate = useNavigate();
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [selectedPatch, setSelectedPatch] = useState(null);
  const logged = useRef(new Set());

  // Auto-select the first finished image; log finished images to history once.
  useEffect(() => {
    items.forEach((it, i) => {
      if (it.status === 'done' && it.result && !logged.current.has(it.result.job_id)) {
        logged.current.add(it.result.job_id);
        history.add({
          kind: 'analyze',
          title: it.name,
          summary: `${VERDICT_SUMMARY[it.result.verdict] || it.result.verdict} · ${Math.round(it.result.confidence * 100)}%`,
          jobId: it.result.job_id,
          result: it.result,
        });
        if (selectedIdx == null) setSelectedIdx(i);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items]);

  // Re-open an analyze result from the history drawer.
  useEffect(() => {
    const r = location.state?.restoreAnalyze;
    if (r) {
      restore(r.result, r.name);
      setSelectedIdx(0);
      setSelectedPatch(null);
      // clear router state so a refresh doesn't re-trigger
      navigate(location.pathname, { replace: true, state: null });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.state]);

  function startRun(files) {
    setSelectedIdx(null);
    setSelectedPatch(null);
    logged.current = new Set();
    run(files);
  }

  function selectCard(i) {
    setSelectedIdx(i);
    setSelectedPatch(null);
  }

  const selected = selectedIdx != null ? items[selectedIdx] : null;
  const result = selected?.result;
  const [sanitizing, setSanitizing] = useState(false);

  async function handleSanitize() {
    if (!result) return;
    setSanitizing(true);
    try { await downloadSanitized(result.job_id, selected.name); }
    finally { setSanitizing(false); }
  }

  return (
    <div className={ui.page}>
      <h1 className={ui.pageTitle}>Analyze images</h1>
      <p className={ui.pageSub}>
        Upload one or more images to scan for hidden data. Each gets a verdict, a suspicion map, and visual tools.
      </p>

      <DemoBar onPick={(f) => startRun([f])} kind="stego" />

      <BatchUploader onRun={startRun} running={running} />

      {items.length > 0 && (
        <div className={styles.list}>
          {items.map((it, i) => (
            <ImageResultCard
              key={i}
              item={it}
              selected={selectedIdx === i}
              onSelect={() => selectCard(i)}
            />
          ))}
        </div>
      )}

      {result && (
        <div className={styles.detail}>
          {result.verdict !== 'CLEAN' && (
            <div className={styles.sanitizeBar}>
              <span className={styles.sanitizeHint}>
                Suspicious image detected — download a steganography-stripped copy?
              </span>
              <button
                className={styles.sanitizeBtn}
                onClick={handleSanitize}
                disabled={sanitizing}
              >
                {sanitizing ? 'Preparing…' : 'Sanitize & download'}
              </button>
            </div>
          )}
          <Dashboard
            result={result}
            selectedPatch={selectedPatch}
            onCellClick={setSelectedPatch}
          />
          <VisualTools
            jobId={result.job_id}
            flow="analyze"
            windowRows={result.window_rows}
            windowCols={result.window_cols}
            selectedPatch={selectedPatch}
          />
          <BitPlaneExplorer jobId={result.job_id} scores={result.bitplane_scores} flow="analyze" />
        </div>
      )}
    </div>
  );
}
