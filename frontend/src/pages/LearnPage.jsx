import { Link } from 'react-router-dom';
import ui from '../styles/ui.module.css';
import styles from './LearnPage.module.css';
import { METHODS, METHOD_ORDER } from '../config/methods.js';

const STEPS = [
  { n: 1, title: 'Hide', body: 'Pick an image, type a message, optionally encrypt it, and embed it with one of five techniques.', to: '/embed', cta: 'Hide a message' },
  { n: 2, title: 'Reveal', body: 'Upload a modified image and enter the passphrase to recover the original message.', to: '/extract', cta: 'Reveal a message' },
  { n: 3, title: 'Analyze', body: 'Scan any image with the detector to see whether it likely contains hidden data.', to: '/analyze', cta: 'Analyze images' },
];

const GLOSSARY = [
  ['Original', 'The unmodified image, before anything is hidden inside it.'],
  ['Modified', 'An image that has hidden data embedded in it.'],
  ['Pixel-level hiding', 'Stores bits in the least-significant bit of each pixel — invisible to the eye.'],
  ['JPEG frequency hiding', 'Hides data in the frequency coefficients used by JPEG compression.'],
  ['Frequency-domain hiding', 'Hides data across global frequency rings of the image.'],
  ['Adaptive spatial hiding', 'Hides data in noisy, textured regions where changes are hardest to detect.'],
  ['AI-learned hiding', 'A trained neural network decides where to hide data so it blends into the image.'],
  ['Quality score', 'How visually identical the modified image is to the original. Above ~40 dB looks the same.'],
  ['Suspicion score', 'The model’s estimated likelihood that a region contains hidden data.'],
  ['Suspicion map', 'A heat overlay showing which areas the model finds suspicious.'],
  ['What the model sees', 'An amplified residual that exposes the tiny pixel changes the detector keys off.'],
  ['Bit-plane', 'One layer of the image’s bits. The lowest layer (LSB) is where pixel-level hiding lives.'],
];

export default function LearnPage() {
  return (
    <div className={ui.page}>
      <h1 className={ui.pageTitle}>How it works</h1>
      <p className={ui.pageSub}>
        Steganography hides a message inside an ordinary-looking image. This tool lets you hide, reveal,
        and detect such messages — and see exactly what the detector looks at.
      </p>

      <div className={styles.steps}>
        {STEPS.map((s) => (
          <div key={s.n} className={`${ui.card} ${styles.step}`}>
            <span className={styles.num}>{s.n}</span>
            <h3 className={styles.stepTitle}>{s.title}</h3>
            <p className={styles.stepBody}>{s.body}</p>
            <Link to={s.to} className={styles.link}>{s.cta} →</Link>
          </div>
        ))}
      </div>

      <h2 className={styles.h2}>The five hiding techniques</h2>
      <div className={styles.methods}>
        {METHOD_ORDER.map((id) => {
          const m = METHODS[id];
          return (
            <div key={id} className={`${ui.card} ${styles.method}`}>
              <div className={styles.methodHead}>
                <span className={styles.methodName}>{m.plain}</span>
                <span className={ui.badge}>{m.name}</span>
              </div>
              <p className={styles.methodBlurb}>{m.blurb}</p>
              <span className={`${styles.recover} ${m.recoverable ? styles.yes : styles.no}`}>
                {m.recoverable ? 'Carries a recoverable message' : 'Statistical noise only'}
              </span>
            </div>
          );
        })}
      </div>

      <h2 className={styles.h2}>Glossary</h2>
      <dl className={styles.glossary}>
        {GLOSSARY.map(([term, def]) => (
          <div key={term} className={styles.term}>
            <dt>{term}</dt>
            <dd>{def}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
