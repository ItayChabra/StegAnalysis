import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import ui from '../styles/ui.module.css';
import styles from './ExtractPage.module.css';
import FilePicker from '../components/FilePicker/FilePicker.jsx';
import ParamControls from '../components/Embed/ParamControls.jsx';
import DemoBar from '../components/Demo/DemoBar.jsx';
import useExtract from '../hooks/useExtract.js';
import { useApp } from '../context/AppContext.jsx';
import { METHODS, defaultParams } from '../config/methods.js';

const EXTRACT_METHODS = ['lsb', 'dct', 'fft'];

export default function ExtractPage() {
  const location = useLocation();
  const { history } = useApp();
  const { state, result, error, extract } = useExtract();

  const incoming = location.state || {};
  const [file, setFile] = useState(incoming.file || null);
  const [passphrase, setPassphrase] = useState('');
  const [advanced, setAdvanced] = useState(false);
  const [method, setMethod] = useState('auto');   // 'auto' | 'lsb' | 'dct' | 'fft'
  const [params, setParams] = useState(null);
  const [copied, setCopied] = useState(false);

  // Pre-fill method + params when arriving from the Hide page.
  useEffect(() => {
    const hint = incoming.hint;
    if (hint && EXTRACT_METHODS.includes(hint.method)) {
      setAdvanced(true);
      setMethod(hint.method);
      const base = defaultParams(hint.method);
      for (const k of Object.keys(base)) if (hint[k] !== undefined) base[k] = hint[k];
      setParams(base);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function changeMethod(m) {
    setMethod(m);
    setParams(m === 'auto' ? null : defaultParams(m));
  }

  const busy = state === 'PROCESSING';

  async function onSubmit() {
    const opts = { passphrase };
    if (advanced && method !== 'auto') {
      opts.method = method;
      Object.assign(opts, params || {});
    }
    try {
      const data = await extract(file, opts);
      history.add({
        kind: 'extract',
        title: `${METHODS[data.method]?.plain || data.method} · ${file.name}`,
        summary: `${data.bytes} bytes · ${data.encrypted ? data.cipher : 'plaintext'}`,
        meta: { message: data.message, method: data.method, cipher: data.cipher, bytes: data.bytes },
      });
    } catch { /* surfaced via error */ }
  }

  function copy() {
    navigator.clipboard.writeText(result.message);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  function downloadText() {
    const blob = new Blob([result.message], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'recovered_message.txt';
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className={ui.page}>
      <h1 className={ui.pageTitle}>Reveal a hidden message</h1>
      <p className={ui.pageSub}>
        Upload a modified image and enter the passphrase to recover the hidden message.
      </p>

      <DemoBar onPick={setFile} kind="reveal" />

      <div className={ui.grid2}>
        {/* ── Inputs ── */}
        <div className={ui.card}>
          <h3 className={ui.cardTitle}>Image</h3>
          <FilePicker file={file} onFile={setFile} label="Drop the modified image" compact />

          <hr className={ui.divider} />
          <div className={ui.field}>
            <label className={ui.label} htmlFor="pp">Passphrase</label>
            <input
              id="pp" type="password" className={ui.input}
              value={passphrase} placeholder="Leave blank if it wasn't encrypted"
              onChange={(e) => setPassphrase(e.target.value)}
              autoComplete="off"
            />
          </div>

          <button className={ui.btnGhost + ' ' + ui.btn} onClick={() => setAdvanced((v) => !v)} style={{ width: '100%' }}>
            {advanced ? '▾ Advanced settings' : '▸ Advanced settings'}
          </button>

          {advanced && (
            <div style={{ marginTop: 14 }}>
              <div className={ui.field}>
                <label className={ui.label} htmlFor="m">Method</label>
                <select id="m" className={ui.select} value={method} onChange={(e) => changeMethod(e.target.value)}>
                  <option value="auto">Auto-detect</option>
                  {EXTRACT_METHODS.map((m) => (
                    <option key={m} value={m}>{METHODS[m].plain} ({METHODS[m].name})</option>
                  ))}
                </select>
                <p className={ui.hint}>Auto-detect works for default settings. If you changed strength/step on embed, pick the method and match them here.</p>
              </div>
              {method !== 'auto' && params && (
                <ParamControls methodId={method} values={params} onChange={(k, v) => setParams((p) => ({ ...p, [k]: v }))} />
              )}
            </div>
          )}

          <button className={ui.btnPrimary} disabled={!file || busy} onClick={onSubmit} style={{ marginTop: 14 }}>
            {busy ? <><span className={ui.spinner} /> Revealing…</> : 'Reveal message →'}
          </button>
        </div>

        {/* ── Result ── */}
        <div className={ui.card}>
          <h3 className={ui.cardTitle}>Recovered message</h3>

          {state === 'ERROR' && (
            <div className={styles.errorBox}>
              <span className={styles.errIcon}>{error?.code === 'bad_key' ? '🔒' : '∅'}</span>
              <div>
                <strong className={styles.errTitle}>
                  {error?.code === 'bad_key' ? 'Wrong passphrase'
                    : error?.code === 'no_payload' ? 'No hidden message found'
                    : 'Could not reveal'}
                </strong>
                <p className={styles.errMsg}>
                  {error?.code === 'bad_key'
                    ? 'The passphrase is incorrect, or the image was altered.'
                    : error?.code === 'no_payload'
                    ? 'No recoverable message in this image. Check the method/settings used to hide it.'
                    : error?.message}
                </p>
              </div>
            </div>
          )}

          {state === 'COMPLETE' && result && (
            <div className={styles.result}>
              <div className={styles.meta}>
                <span className={ui.badge}>{METHODS[result.method]?.plain || result.method}</span>
                <span className={ui.badge}>{result.encrypted ? result.cipher : 'plaintext'}</span>
                <span className={ui.badge}>{result.bytes} bytes</span>
              </div>
              <pre className={styles.message}>{result.message}</pre>
              <div className={ui.row}>
                <button className={ui.btn} onClick={copy}>{copied ? '✓ Copied' : 'Copy'}</button>
                <button className={ui.btn} onClick={downloadText}>↓ Download .txt</button>
              </div>
            </div>
          )}

          {state === 'IDLE' && <p className={ui.hint}>The recovered message will appear here.</p>}
          {busy && <p className={ui.hint}>Scanning the image…</p>}
        </div>
      </div>
    </div>
  );
}
