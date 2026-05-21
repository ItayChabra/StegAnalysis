import styles         from './Dashboard.module.css';
import VerdictBanner   from './VerdictBanner.jsx';
import ConfidenceMeter from './ConfidenceMeter.jsx';
import WindowHeatGrid  from './WindowHeatGrid.jsx';
import StatsRow        from './StatsRow.jsx';
import MethodBadge     from './MethodBadge.jsx';

// selectedPatch (flat index | null) and onCellClick live in App so the
// ImageComparisonSlider can also read selectedPatch without prop-drilling
// through Dashboard.
export default function Dashboard({ result, selectedPatch, onCellClick }) {
  return (
    <section className={styles.dashboard} aria-label="Analysis results">
      <VerdictBanner verdict={result.verdict} />

      <ConfidenceMeter confidence={result.confidence} />

      <WindowHeatGrid
        windowRows={result.window_rows}
        windowCols={result.window_cols}
        windowScores={result.window_scores}
        onCellClick={onCellClick}
        selectedIndex={selectedPatch}
      />

      <StatsRow
        flaggedWindows={result.flagged_windows}
        totalWindows={result.total_windows}
        maxWindowScore={result.max_window_score}
        psnr={result.psnr}
      />

      <MethodBadge methodHint={result.method_hint} />
    </section>
  );
}
