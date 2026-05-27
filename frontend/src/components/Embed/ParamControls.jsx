import ui from '../../styles/ui.module.css';
import { METHODS } from '../../config/methods.js';

// Renders the hyperparameter controls for a method, driven by config/methods.js.
export default function ParamControls({ methodId, values, onChange }) {
  const method = METHODS[methodId];
  return (
    <div className={ui.row}>
      {method.params.map((p) => {
        if (p.showIf && !p.showIf(values)) return null;
        return (
          <div className={ui.field} key={p.key}>
            <label className={ui.label} htmlFor={`p_${p.key}`}>{p.label}</label>
            {p.type === 'select' ? (
              <select
                id={`p_${p.key}`}
                className={ui.select}
                value={values[p.key]}
                onChange={(e) => onChange(p.key, e.target.value)}
              >
                {p.options.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            ) : (
              <input
                id={`p_${p.key}`}
                type="number"
                className={ui.input}
                value={values[p.key]}
                min={p.min} max={p.max} step={p.step}
                onChange={(e) => onChange(p.key, e.target.value === '' ? '' : Number(e.target.value))}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
