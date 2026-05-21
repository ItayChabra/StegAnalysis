import { useState, useEffect } from 'react';
import styles from './ConfidenceMeter.module.css';

function tierClass(confidence) {
  if (confidence > 0.75) return styles.high;
  if (confidence >= 0.5) return styles.mid;
  return styles.low;
}

export default function ConfidenceMeter({ confidence }) {
  const [filled, setFilled] = useState(false);
  const pct = Math.round(confidence * 100);

  useEffect(() => {
    const id = requestAnimationFrame(() => setFilled(true));
    return () => cancelAnimationFrame(id);
  }, []);

  return (
    <div className={styles.wrapper}>
      <div className={styles.header}>
        <p className={styles.heading}>Model Confidence</p>
        <p className={styles.pct}>{pct}%</p>
      </div>
      <div className={styles.track}>
        <div
          className={`${styles.fill} ${tierClass(confidence)}`}
          style={{ width: filled ? `${pct}%` : '0%' }}
        />
      </div>
    </div>
  );
}
