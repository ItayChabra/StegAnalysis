import styles from './GeneratorPicker.module.css';
import { METHODS, METHOD_ORDER } from '../../config/methods.js';

export default function GeneratorPicker({ value, onChange }) {
  return (
    <div className={styles.grid} role="radiogroup" aria-label="Hiding method">
      {METHOD_ORDER.map((id) => {
        const m = METHODS[id];
        const active = value === id;
        return (
          <button
            key={id}
            role="radio"
            aria-checked={active}
            className={`${styles.card}${active ? ` ${styles.active}` : ''}`}
            onClick={() => onChange(id)}
          >
            <div className={styles.top}>
              <span className={styles.plain}>{m.plain}</span>
              {!m.recoverable && <span className={styles.tag}>noise only</span>}
              {m.fragile && <span className={styles.tagWarn}>fragile</span>}
            </div>
            <span className={styles.tech}>{m.name}</span>
            <span className={styles.blurb}>{m.blurb}</span>
          </button>
        );
      })}
    </div>
  );
}
