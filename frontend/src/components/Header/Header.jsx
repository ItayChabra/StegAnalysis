import { NavLink } from 'react-router-dom';
import styles from './Header.module.css';
import { useApp } from '../../context/AppContext.jsx';

const NAV = [
  { to: '/analyze', label: 'Analyze' },
  { to: '/embed',   label: 'Hide' },
  { to: '/extract', label: 'Reveal' },
  { to: '/learn',   label: 'Learn' },
];

export default function Header() {
  const { openHistory, history } = useApp();

  return (
    <header className={styles.header}>
      <div className={styles.left}>
        <div className={styles.logoMark} aria-hidden="true" />
        <span className={styles.wordmark}>SRNet Steganalysis</span>
      </div>

      <nav className={styles.nav} aria-label="Primary">
        {NAV.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `${styles.navLink}${isActive ? ` ${styles.navActive}` : ''}`}
          >
            {label}
          </NavLink>
        ))}
      </nav>

      <button
        className={styles.historyBtn}
        onClick={openHistory}
        aria-label="Open session history"
      >
        History
        {history.entries.length > 0 && (
          <span className={styles.count}>{history.entries.length}</span>
        )}
      </button>
    </header>
  );
}
