import styles from './WindowHeatGrid.module.css';

function getScoreTier(score) {
  if (score >= 0.90) return '4';
  if (score >= 0.75) return '3';
  if (score >= 0.50) return '2';
  if (score >= 0.25) return '1';
  return '0';
}

function scoreToColor(score) {
  const hue = 220 - Math.round(score * 220);
  const sat = 55  + Math.round(score * 35);
  const lit = 30  + Math.round(score * 25);
  return `hsl(${hue}, ${sat}%, ${lit}%)`;
}

const LEGEND = [
  { tier: '0', label: 'Clean',      color: '#0f1118' },
  { tier: '1', label: 'Low',        color: scoreToColor(0.35) },
  { tier: '2', label: 'Moderate',   color: scoreToColor(0.60) },
  { tier: '3', label: 'High',       color: scoreToColor(0.82) },
  { tier: '4', label: 'Critical',   color: scoreToColor(0.95) },
];

export default function WindowHeatGrid({
  windowRows,
  windowCols,
  windowScores,
  onCellClick,
  selectedIndex,
}) {
  if (!windowScores || windowScores.length === 0) return null;

  return (
    <div className={styles.wrapper}>
      <p className={styles.heading}>Patch Analysis Grid</p>

      <div
        className={styles.grid}
        style={{ gridTemplateColumns: `repeat(${windowCols}, 1fr)` }}
      >
        {windowScores.map((score, i) => (
          <div
            key={i}
            className={`${styles.cell}${selectedIndex === i ? ` ${styles.selected}` : ''}`}
            data-tier={getScoreTier(score)}
            data-score={`${Math.round(score * 100)}%`}
            style={{
              backgroundColor:  scoreToColor(score),
              animationDelay:   `${i * 20}ms`,
            }}
            onClick={() => onCellClick?.(i)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') onCellClick?.(i);
            }}
            aria-label={`Patch ${i}: ${Math.round(score * 100)}% suspicion`}
            aria-pressed={selectedIndex === i}
          />
        ))}
      </div>

      <div className={styles.legend} aria-hidden="true">
        {LEGEND.map(({ tier, label, color }) => (
          <span key={tier} className={styles.legendItem}>
            <span
              className={styles.legendSwatch}
              style={{ background: color }}
            />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
