import styles from './ErrorBanner.module.css';

export default function ErrorBanner({ error, reset }) {
  return (
    <div className={styles.banner} role="alert">
      <div className={styles.body}>
        <p className={styles.title}>⚠ Something went wrong</p>
        <p className={styles.message}>{error?.message ?? 'An unexpected error occurred.'}</p>
      </div>
      <button className={styles.retryBtn} onClick={reset}>
        Try again
      </button>
    </div>
  );
}
