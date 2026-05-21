import styles from './MethodBadge.module.css';

const METHOD_LABELS = {
  lsb_sequential: 'Pixel-level hiding',
  lsb_edge:       'Pixel-level hiding',
  lsb_random:     'Pixel-level hiding',
  dct_mid:        'JPEG frequency hiding',
  dct_low_mid:    'JPEG frequency hiding',
  fft_mid:        'Frequency-domain hiding',
  fft_high:       'Frequency-domain hiding',
  fft_low:        'Frequency-domain hiding',
};

export default function MethodBadge({ methodHint }) {
  if (!methodHint) return null;

  const label = METHOD_LABELS[methodHint] ?? methodHint;

  return (
    <div className={styles.badge}>
      <span className={styles.dot} aria-hidden="true" />
      <p className={styles.text}>Pattern matches: {label}</p>
    </div>
  );
}
