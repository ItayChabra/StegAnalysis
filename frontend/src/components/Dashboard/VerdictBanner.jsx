import { useState, useEffect } from 'react';
import styles from './VerdictBanner.module.css';

const VERDICT_MAP = {
  CLEAN:          { label: 'CLEAN',             icon: '✓', cls: styles.clean },
  SUSPICIOUS:     { label: 'SUSPICIOUS',        icon: '⚠', cls: styles.warn  },
  STEGO_DETECTED: { label: 'HIDDEN DATA FOUND', icon: '🔍', cls: styles.alert },
};

export default function VerdictBanner({ verdict }) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const id = requestAnimationFrame(() => setVisible(true));
    return () => cancelAnimationFrame(id);
  }, []);

  const cfg = VERDICT_MAP[verdict] ?? VERDICT_MAP.CLEAN;

  return (
    <div className={`${styles.banner} ${cfg.cls}${visible ? ` ${styles.visible}` : ''}`}>
      <div className={styles.iconWrap} aria-hidden="true">{cfg.icon}</div>
      <div className={styles.text}>
        <p className={styles.label}>Analysis Result</p>
        <p className={styles.verdict}>{cfg.label}</p>
      </div>
    </div>
  );
}
