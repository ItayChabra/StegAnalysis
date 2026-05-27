// Build and download a per-image JSON report from an analyze result.
export function buildReport(result) {
  return {
    filename: result.filename,
    verdict: result.verdict,
    confidence: result.confidence,
    max_window_score: result.max_window_score,
    mean_window_score: result.mean_window_score,
    flagged_windows: result.flagged_windows,
    total_windows: result.total_windows,
    window_grid: { rows: result.window_rows, cols: result.window_cols },
    detector_kind: 'Binary clean/stego (the model does not classify embedding method).',
    bitplane_scores: result.bitplane_scores,
    window_scores: result.window_scores,
    generated_at: new Date().toISOString(),
  };
}

export function downloadReport(result) {
  const blob = new Blob([JSON.stringify(buildReport(result), null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const base = (result.filename || 'image').replace(/\.[^.]+$/, '');
  a.href = url;
  a.download = `report_${base}.json`;
  a.click();
  URL.revokeObjectURL(url);
}
