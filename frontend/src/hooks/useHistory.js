import { useCallback, useEffect, useState } from 'react';

// Session analysis/embed/extract log, stored in sessionStorage (clears on tab close).
// Entries: { id, ts, kind: 'analyze'|'embed'|'extract', title, summary, meta }

const KEY = 'srnet.history.v1';
const MAX = 50;

function load() {
  try {
    return JSON.parse(sessionStorage.getItem(KEY)) || [];
  } catch {
    return [];
  }
}

export default function useHistory() {
  const [entries, setEntries] = useState(load);

  useEffect(() => {
    try { sessionStorage.setItem(KEY, JSON.stringify(entries)); } catch { /* quota */ }
  }, [entries]);

  const add = useCallback((entry) => {
    const created = { id: crypto.randomUUID(), ts: Date.now(), ...entry };
    setEntries((prev) => [created, ...prev].slice(0, MAX));
    return created;
  }, []);

  const updateMeta = useCallback((id, patch) => {
    setEntries((prev) => prev.map((e) =>
      e.id === id ? { ...e, meta: { ...e.meta, ...patch } } : e
    ));
  }, []);

  const remove = useCallback((id) => {
    setEntries((prev) => prev.filter((e) => e.id !== id));
  }, []);

  const clear = useCallback(() => setEntries([]), []);

  return { entries, add, updateMeta, remove, clear };
}
