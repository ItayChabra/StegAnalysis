import { useCallback, useRef, useState } from 'react';
import { analyzeImage } from '../api/client.js';

// Batch analyze: runs /api/analyze per file with a small concurrency cap and
// reports per-file progress. items: [{ file, name, status, result, error }]
const CONCURRENCY = 2;

export default function useBatchAnalysis() {
  const [items, setItems]   = useState([]);
  const [running, setRunning] = useState(false);
  const cancelled = useRef(false);

  const update = useCallback((idx, patch) => {
    setItems((prev) => prev.map((it, i) => (i === idx ? { ...it, ...patch } : it)));
  }, []);

  const run = useCallback(async (files) => {
    cancelled.current = false;
    const initial = Array.from(files).map((file) => ({
      file, name: file.name, status: 'pending', result: null, error: null,
    }));
    setItems(initial);
    setRunning(true);

    let cursor = 0;
    const worker = async () => {
      while (cursor < initial.length && !cancelled.current) {
        const idx = cursor++;
        update(idx, { status: 'running' });
        try {
          const result = await analyzeImage(initial[idx].file);
          update(idx, { status: 'done', result });
        } catch (err) {
          update(idx, { status: 'error', error: err });
        }
      }
    };

    await Promise.all(
      Array.from({ length: Math.min(CONCURRENCY, initial.length) }, worker)
    );
    setRunning(false);
  }, [update]);

  const reset = useCallback(() => {
    cancelled.current = true;
    setItems([]);
    setRunning(false);
  }, []);

  // Insert a previously-completed analysis (from the history drawer) so the
  // page can show its dashboard without re-running inference. The server must
  // still hold the job's artefacts (heatmap / noisemap / etc.) for the
  // visual tools to render — see HistoryDrawer's "reopen" handler.
  const restore = useCallback((result, name) => {
    if (!result) return;
    cancelled.current = true;
    setItems([{ file: null, name: name || result.filename || 'restored', status: 'done', result, error: null }]);
    setRunning(false);
  }, []);

  return { items, running, run, reset, restore };
}
