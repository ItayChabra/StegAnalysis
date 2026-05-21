import { useState } from 'react';
import styles from './Header.module.css';

export default function Header() {
  const [demoMode, setDemoMode] = useState(false);

  return (
    <header className={styles.header}>
      <div className={styles.left}>
        <div className={styles.logoMark} aria-hidden="true" />
        <span className={styles.wordmark}>SRNet Steganalysis</span>
      </div>
      <button
        className={`${styles.demoToggle}${demoMode ? ` ${styles.on}` : ''}`}
        onClick={() => setDemoMode(v => !v)}
        aria-pressed={demoMode}
      >
        Demo Mode
      </button>
    </header>
  );
}
