import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ui from '../styles/ui.module.css';
import styles from './EmbedPage.module.css';
import FilePicker from '../components/FilePicker/FilePicker.jsx';
import GeneratorPicker from '../components/Embed/GeneratorPicker.jsx';
import ParamControls from '../components/Embed/ParamControls.jsx';
import EncryptionControls from '../components/Embed/EncryptionControls.jsx';
import CapacityMeter from '../components/Embed/CapacityMeter.jsx';
import VisualTools from '../components/Visual/VisualTools.jsx';
import BitPlaneExplorer from '../components/Visual/BitPlaneExplorer.jsx';
import DemoBar from '../components/Demo/DemoBar.jsx';
import useEmbed from '../hooks/useEmbed.js';
import useCapacity from '../hooks/useCapacity.js';
import { useApp } from '../context/AppContext.jsx';
import { METHODS, defaultParams } from '../config/methods.js';
import { stegoUrl } from '../api/client.js';

async function downloadStego(jobId) {
  const res = await fetch(stegoUrl(jobId));
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `stego_${jobId.slice(0, 8)}.png`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function EmbedPage() {
  const navigate = useNavigate();
  const { history } = useApp();
  const { state, result, error, embed } = useEmbed();
  const capacity = useCapacity();

  const [file, setFile] = useState(null);
  const [methodId, setMethodId] = useState('lsb');
  const [params, setParams] = useState(() => defaultParams('lsb'));
  const [message, setMessage] = useState('');
  const [cipher, setCipher] = useState('none');
  const [passphrase, setPassphrase] = useState('');

  const method = METHODS[methodId];
  const messageBytes = useMemo(() => new TextEncoder().encode(message).length, [message]);

  function changeMethod(id) {
    setMethodId(id);
    setParams(defaultParams(id));
  }

  function changeParam(key, value) {
    setParams((p) => ({ ...p, [key]: value }));
  }

  // Live capacity estimate for recoverable methods.
  useEffect(() => {
    if (file && method.recoverable) capacity.estimate(file, methodId, params);
    else capacity.clear();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file, methodId, JSON.stringify(params)]);

  const busy = state === 'PROCESSING';
  const canSubmit = file && !busy && (method.recoverable ? message.length > 0 : true)
    && (cipher === 'none' || passphrase.length > 0);

  async function onSubmit() {
    const opts = { method: methodId, ...params };
    if (method.recoverable) {
      opts.message = message;
      opts.cipher = cipher;
      if (cipher !== 'none') opts.passphrase = passphrase;
    }
    try {
      const data = await embed(file, opts);
      history.add({
        kind: 'embed',
        title: `${method.plain} · ${file.name}`,
        summary: data.recoverable
          ? `${data.cipher === 'none' ? 'plaintext' : data.cipher} · PSNR ${data.psnr ?? '—'} dB`
          : 'noise only',
        jobId: data.job_id,
        meta: { hint: data.extract_hint, recoverable: data.recoverable, psnr: data.psnr, cipher: data.cipher },
      });
    } catch { /* surfaced via error */ }
  }

  async function revealThis() {
    const res = await fetch(stegoUrl(result.job_id));
    const blob = await res.blob();
    const f = new File([blob], `stego_${result.job_id.slice(0, 8)}.png`, { type: 'image/png' });
    navigate('/extract', { state: { file: f, hint: result.extract_hint } });
  }

  return (
    <div className={ui.page}>
      <h1 className={ui.pageTitle}>Hide a message</h1>
      <p className={ui.pageSub}>
        Embed a secret message inside an image. Optionally encrypt it first, then download the result.
      </p>

      <DemoBar onPick={(f) => setFile(f)} kind="cover" />

      <div className={ui.grid2}>
        {/* ── Left: inputs ── */}
        <div className={ui.card}>
          <h3 className={ui.cardTitle}>1 · Image</h3>
          <FilePicker file={file} onFile={setFile} label="Drop a cover image" compact />

          <hr className={ui.divider} />
          <h3 className={ui.cardTitle}>2 · Method</h3>
          <GeneratorPicker value={methodId} onChange={changeMethod} />
          <div style={{ marginTop: 14 }}>
            <ParamControls methodId={methodId} values={params} onChange={changeParam} />
          </div>

          {method.recoverable && (
            <>
              <hr className={ui.divider} />
              <h3 className={ui.cardTitle}>3 · Message</h3>
              <div className={ui.field}>
                <textarea
                  className={ui.textarea}
                  value={message}
                  placeholder="Type the secret message to hide…"
                  onChange={(e) => setMessage(e.target.value)}
                />
              </div>
              <CapacityMeter data={capacity.data} loading={capacity.loading} messageBytes={messageBytes} />

              <hr className={ui.divider} />
              <h3 className={ui.cardTitle}>4 · Encryption</h3>
              <EncryptionControls
                cipher={cipher} passphrase={passphrase}
                onCipher={setCipher} onPassphrase={setPassphrase}
              />
            </>
          )}

          <button className={ui.btnPrimary} disabled={!canSubmit} onClick={onSubmit} style={{ marginTop: 8 }}>
            {busy ? <><span className={ui.spinner} /> Hiding…</> : 'Hide message →'}
          </button>
          {error && <p className={ui.errorText}>{error.message}</p>}
        </div>

        {/* ── Right: result ── */}
        <div className={ui.card}>
          <h3 className={ui.cardTitle}>Result</h3>
          {state !== 'COMPLETE' || !result ? (
            <p className={ui.hint}>The modified image and its quality score will appear here.</p>
          ) : (
            <div className={styles.result}>
              <img src={stegoUrl(result.job_id)} alt="Modified image" className={styles.stego} />
              <div className={styles.stats}>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Quality score</span>
                  <span className={styles.statValue}>{result.psnr != null ? `${result.psnr} dB` : '—'}</span>
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Method</span>
                  <span className={styles.statValue}>{METHODS[result.method]?.plain || result.method}</span>
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Encryption</span>
                  <span className={styles.statValue}>{result.cipher === 'none' ? 'None' : result.cipher}</span>
                </div>
              </div>
              <div className={ui.row}>
                <button className={ui.btn} onClick={() => downloadStego(result.job_id)}>↓ Download image</button>
                {result.recoverable && (
                  <button className={ui.btn} onClick={revealThis}>Reveal message →</button>
                )}
              </div>
              {!result.recoverable && result.note && <p className={ui.hint}>{result.note}</p>}
            </div>
          )}
        </div>
      </div>

      {state === 'COMPLETE' && result && (
        <div className={styles.tools}>
          <VisualTools jobId={result.job_id} flow="embed" />
          <BitPlaneExplorer jobId={result.job_id} scores={result.bitplane_scores} flow="embed" />
        </div>
      )}
    </div>
  );
}
