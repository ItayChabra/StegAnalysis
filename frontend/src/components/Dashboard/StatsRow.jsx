import styles from './StatsRow.module.css';

export default function StatsRow({
  flaggedWindows,
  totalWindows,
  maxWindowScore,
  psnr,
}) {
  const peakPct   = `${Math.round(maxWindowScore * 100)}%`;
  const qualScore = psnr != null ? `${psnr.toFixed(1)} dB` : 'N/A';
  const flagRatio = totalWindows > 0
    ? `${flaggedWindows} / ${totalWindows}`
    : '— / —';

  return (
    <div className={styles.row}>
      <div className={styles.card}>
        <p className={styles.cardLabel}>Flagged Patches</p>
        <p className={styles.cardValue}>{flagRatio}</p>
        {totalWindows > 0 && (
          <p className={styles.cardSub}>
            {Math.round((flaggedWindows / totalWindows) * 100)}% of total
          </p>
        )}
      </div>

      <div className={styles.card}>
        <p className={styles.cardLabel}>Peak Suspicion</p>
        <p className={styles.cardValue}>{peakPct}</p>
        <p className={styles.cardSub}>highest patch score</p>
      </div>

      <div className={styles.card}>
        <p className={styles.cardLabel}>Quality Score</p>
        <p className={styles.cardValue}>{qualScore}</p>
        <p className={styles.cardSub}>{psnr != null ? '>40 dB = imperceptible' : 'no reference image'}</p>
      </div>
    </div>
  );
}
