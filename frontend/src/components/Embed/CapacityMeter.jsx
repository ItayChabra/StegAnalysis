import ui from '../../styles/ui.module.css';
import styles from './CapacityMeter.module.css';

// Shows how much of the image's recoverable capacity the current message uses.
export default function CapacityMeter({ data, loading, messageBytes }) {
  if (!data) {
    return <p className={ui.hint}>{loading ? 'Estimating capacity…' : 'Add an image to estimate capacity.'}</p>;
  }
  if (!data.recoverable) {
    return <p className={ui.hint}>This method embeds statistical noise and carries no recoverable message.</p>;
  }

  const max = data.max_message_bytes || 0;
  const used = messageBytes || 0;
  const pct = max > 0 ? Math.min(100, (used / max) * 100) : 0;
  const over = used > max;

  return (
    <div className={styles.wrap}>
      <div className={styles.row}>
        <span className={styles.label}>Capacity</span>
        <span className={`${styles.value}${over ? ` ${styles.over}` : ''}`}>
          {used.toLocaleString()} / {max.toLocaleString()} bytes
        </span>
      </div>
      <div className={styles.track}>
        <div
          className={`${styles.fill}${over ? ` ${styles.fillOver}` : ''}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {over && <p className={ui.errorText}>Message is too large for this image and settings.</p>}
    </div>
  );
}
