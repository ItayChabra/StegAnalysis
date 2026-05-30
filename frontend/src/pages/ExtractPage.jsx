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
  const { state, result, plaintext, error, extract, decrypt } = useExtract();

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

  const revealing  = state === 'PROCESSING';
  const decrypting = state === 'DECRYPTING';
  const revealed   = state === 'REVEALED' || state === 'DECRYPTING' || state === 'DECRYPTED';
  const decrypted  = state === 'DECRYPTED' && plaintext;
  const badKey     = state === 'REVEALED' && error?.code === 'bad_key';

  async function onReveal() {
    setPassphrase('');
    const opts = {};
    if (advanced && method !== 'auto') {
      opts.method = method;
      Object.assign(opts, params || {});
    }
    try {
      const data = await extract(file, opts);
      history.add({
        kind: 'extract',
        title: `${METHODS[data.method]?.plain || data.method} · ${file.name}`,
        summary: data.encrypted
          ? `${data.bytes} bytes · ${data.cipher} (encrypted)`
          : `${data.bytes} bytes · plaintext`,
        meta: {
          method:    data.method,
          cipher:    data.cipher,
          bytes:     data.bytes,
          encrypted: data.encrypted,
          message:   data.encrypted ? null : data.message,
        },
      });
    } catch { /* surfaced via error */ }
  }

  async function onDecrypt() {
    try { await decrypt(passphrase); } catch { /* surfaced via error */ }
  }

  // What we actually show in the message panel — plaintext if available,
  // else the plaintext returned at reveal time (cipher=none), else the
  // base64 ciphertext.
  const displayMessage = decrypted
    ? plaintext.message
    : result?.encrypted
      ? result.ciphertext_b64
      : result?.message;

  const displayKind = decrypted
    ? 'plaintext'
    : result?.encrypted
      ? 'ciphertext'
      : 'plaintext';

  function copy() {
    navigator.clipboard.writeText(displayMessage || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  function downloadText() {
    const blob = new Blob([displayMessage || ''], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = displayKind === 'ciphertext'
      ? 'recovered_ciphertext.b64.txt'
      : 'recovered_message.txt';
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className={ui.page}>
      <h1 className={ui.pageTitle}>Reveal a hidden message</h1>
      <p className={ui.pageSub}>
        Reveal first — see the raw hidden bytes. If they're encrypted, enter the
        key to decrypt them.
      </p>

      <DemoBar onPick={setFile} kind="reveal" />

      <div className={ui.grid2}>
        {/* ── Inputs ── */}
        <div className={ui.card}>
          <h3 className={ui.cardTitle}>Image</h3>
          <FilePicker file={file} onFile={setFile} label="Drop the modified image" compact />

          <hr className={ui.divider} />

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

          <button className={ui.btnPrimary} disabled={!file || revealing} onClick={onReveal} style={{ marginTop: 14 }}>
            {revealing ? <><span className={ui.spinner} /> Revealing…</> : 'Reveal hidden data →'}
          </button>
        </div>

        {/* ── Result ── */}
        <div className={ui.card}>
          <h3 className={ui.cardTitle}>
            {decrypted ? 'Decrypted message' : revealed ? 'Hidden data' : 'Result'}
          </h3>

          {state === 'ERROR' && (
            <div className={styles.errorBox}>
              <span className={styles.errIcon}>{error?.code === 'no_payload' ? '∅' : '!'}</span>
              <div>
                <strong className={styles.errTitle}>
                  {error?.code === 'no_payload' ? 'No hidden message found' : 'Could not reveal'}
                </strong>
                <p className={styles.errMsg}>
                  {error?.code === 'no_payload'
                    ? 'No recoverable message in this image. Check the method/settings used to hide it.'
                    : error?.message}
                </p>
              </div>
            </div>
          )}

          {revealed && result && (
            <div className={styles.result}>
              <div className={styles.meta}>
                <span className={ui.badge}>{METHODS[result.method]?.plain || result.method}</span>
                <span className={ui.badge}>
                  {result.encrypted ? `${result.cipher} (encrypted)` : 'plaintext'}
                </span>
                <span className={ui.badge}>{result.bytes} bytes</span>
                {decrypted && <span className={ui.badge}>decrypted</span>}
              </div>

              {result.encrypted && !decrypted && (
                <p className={ui.hint} style={{ margin: 0 }}>
                  These are the encrypted bytes (base64). Enter the key below to
                  reveal the actual message.
                </p>
              )}

              <pre className={styles.message}>{displayMessage}</pre>

              <div className={ui.row}>
                <button className={ui.btn} onClick={copy}>{copied ? '✓ Copied' : 'Copy'}</button>
                <button className={ui.btn} onClick={downloadText}>↓ Download .txt</button>
              </div>

              {result.encrypted && !decrypted && (
                <>
                  <hr className={ui.divider} />
                  <div className={ui.field}>
                    <label className={ui.label} htmlFor="pp">Decryption key</label>
                    <input
                      id="pp" type="password" className={ui.input}
                      value={passphrase} placeholder="Enter the passphrase used at embed"
                      onChange={(e) => setPassphrase(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter' && passphrase && !decrypting) onDecrypt(); }}
                      autoComplete="off"
                      autoFocus
                    />
                  </div>

                  {badKey && (
                    <div className={styles.errorBox}>
                      <span className={styles.errIcon}>🔒</span>
                      <div>
                        <strong className={styles.errTitle}>Wrong key</strong>
                        <p className={styles.errMsg}>
                          The key is incorrect, or the image was altered after embedding.
                        </p>
                      </div>
                    </div>
                  )}

                  <button
                    className={ui.btnPrimary}
                    disabled={!passphrase || decrypting}
                    onClick={onDecrypt}
                  >
                    {decrypting ? <><span className={ui.spinner} /> Decrypting…</> : 'Decrypt →'}
                  </button>
                </>
              )}
            </div>
          )}

          {state === 'IDLE' && <p className={ui.hint}>The hidden data will appear here.</p>}
          {revealing && <p className={ui.hint}>Scanning the image…</p>}
        </div>
      </div>
    </div>
  );
}