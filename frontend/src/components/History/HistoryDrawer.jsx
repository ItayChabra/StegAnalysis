import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../../context/AppContext.jsx';
import { originalUrl, stegoUrl } from '../../api/client.js';
import { formatHint } from '../../config/methods.js';
import styles from './HistoryDrawer.module.css';

const KIND_LABEL = { analyze: 'Analyze', embed: 'Hide', extract: 'Reveal' };

function timeAgo(ts) {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(ts).toLocaleDateString();
}

export default function HistoryDrawer() {
  const { historyOpen, closeHistory, history } = useApp();
  const { entries, remove, clear } = history;
  const navigate = useNavigate();
  const [expanded, setExpanded] = useState(null); // entry.id

  function reopen(entry) {
    closeHistory();
    if (entry.kind === 'analyze' && entry.result) {
      navigate('/analyze', { state: { restoreAnalyze: { result: entry.result, name: entry.title } } });
    } else if (entry.kind === 'embed') {
      // Embed entries show inline detail (download/reveal links); just close.
      // The user already has the download in the drawer below.
    } else if (entry.kind === 'extract') {
      // Extract entries show their message inline.
    }
  }

  async function copyMessage(msg) {
    try { await navigator.clipboard.writeText(msg); } catch { /* ignore */ }
  }

  function downloadFromUrl(url, filename) {
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.target = '_blank'; a.click();
  }

  return (
    <>
      <div
        className={`${styles.scrim}${historyOpen ? ` ${styles.scrimOpen}` : ''}`}
        onClick={closeHistory}
        aria-hidden={!historyOpen}
      />
      <aside
        className={`${styles.drawer}${historyOpen ? ` ${styles.drawerOpen}` : ''}`}
        aria-label="Session history"
        aria-hidden={!historyOpen}
      >
        <div className={styles.head}>
          <h2 className={styles.title}>Session History</h2>
          <button className={styles.close} onClick={closeHistory} aria-label="Close history">×</button>
        </div>

        {entries.length === 0 ? (
          <p className={styles.empty}>No activity yet. Analyze, hide, or reveal something to log it here.</p>
        ) : (
          <>
            <button className={styles.clear} onClick={clear}>Clear all</button>
            <ul className={styles.list}>
              {entries.map((e) => {
                const isOpen = expanded === e.id;
                const canReopen = e.kind === 'analyze' && !!e.result;
                const canExpand = e.kind === 'embed' || e.kind === 'extract';
                return (
                  <li key={e.id} className={styles.item}>
                    <div className={styles.itemHead}>
                      <span className={`${styles.kind} ${styles[`kind_${e.kind}`] || ''}`}>
                        {KIND_LABEL[e.kind] || e.kind}
                      </span>
                      <span className={styles.time}>{timeAgo(e.ts)}</span>
                      <button
                        className={styles.remove}
                        onClick={() => remove(e.id)}
                        aria-label="Remove entry"
                      >×</button>
                    </div>
                    <div className={styles.itemTitle}>{e.title}</div>
                    {e.summary && <div className={styles.itemSummary}>{e.summary}</div>}

                    {/* Thumbnail for analyze / embed entries (server may have purged the job) */}
                    {e.jobId && (e.kind === 'analyze' || e.kind === 'embed') && (
                      <img
                        className={styles.thumb}
                        src={e.kind === 'embed' ? stegoUrl(e.jobId) : originalUrl(e.jobId)}
                        alt=""
                        onError={(ev) => { ev.target.style.display = 'none'; }}
                      />
                    )}

                    <div className={styles.actions}>
                      {canReopen && (
                        <button className={styles.action} onClick={() => reopen(e)}>
                          ↗ Re-open
                        </button>
                      )}
                      {e.kind === 'embed' && e.jobId && e.meta?.recoverable && (
                        <button
                          className={styles.action}
                          onClick={() => downloadFromUrl(stegoUrl(e.jobId), `stego_${e.jobId.slice(0,8)}.png`)}
                        >
                          ⬇ Download
                        </button>
                      )}
                      {canExpand && (
                        <button
                          className={styles.action}
                          onClick={() => setExpanded(isOpen ? null : e.id)}
                          aria-expanded={isOpen}
                        >
                          {isOpen ? '▲ Hide details' : '▼ Show details'}
                        </button>
                      )}
                    </div>

                    {/* Expanded details for extract */}
                    {isOpen && e.kind === 'extract' && (
                      <div className={styles.expand}>
                        {e.meta?.message ? (
                          <>
                            <div className={styles.expandLabel}>Recovered message</div>
                            <pre className={styles.message}>{e.meta.message}</pre>
                            <button className={styles.action} onClick={() => copyMessage(e.meta.message)}>
                              📋 Copy
                            </button>
                          </>
                        ) : (
                          <>
                            <div className={styles.expandLabel}>Encrypted payload</div>
                            <p className={styles.message}>
                              {e.meta?.bytes ?? '?'} bytes · {e.meta?.cipher || 'unknown cipher'}.
                              The plaintext isn&apos;t stored in history — re-run Reveal on the
                              image and enter the key to decrypt it.
                            </p>
                          </>
                        )}
                      </div>
                    )}
                    {/* Expanded details for embed */}
                    {isOpen && e.kind === 'embed' && e.meta?.hint && (
                      <div className={styles.expand}>
                        <div className={styles.expandLabel}>Hidden using:</div>
                        <p className={styles.message}>{formatHint(e.meta.hint)}</p>
                        <div className={styles.expandLabel} style={{ marginTop: 8 }}>To reveal:</div>
                        <p className={styles.message}>Upload the modified image on the Reveal page — settings are filled in automatically.</p>
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          </>
        )}
      </aside>
    </>
  );
}
