import ui from '../../styles/ui.module.css';
import { CIPHERS } from '../../config/methods.js';

export default function EncryptionControls({ cipher, passphrase, onCipher, onPassphrase }) {
  const encrypted = cipher !== 'none';
  return (
    <div>
      <div className={ui.field}>
        <label className={ui.label} htmlFor="cipher">Encryption</label>
        <select
          id="cipher"
          className={ui.select}
          value={cipher}
          onChange={(e) => onCipher(e.target.value)}
        >
          {CIPHERS.map((c) => (
            <option key={c.id} value={c.id}>{c.label}</option>
          ))}
        </select>
      </div>

      {encrypted && (
        <div className={ui.field}>
          <label className={ui.label} htmlFor="passphrase">Passphrase</label>
          <input
            id="passphrase"
            type="password"
            className={ui.input}
            value={passphrase}
            placeholder="Required to reveal the message later"
            onChange={(e) => onPassphrase(e.target.value)}
            autoComplete="new-password"
          />
          <p className={ui.hint}>Keep this safe — the message can't be recovered without it.</p>
        </div>
      )}
    </div>
  );
}
