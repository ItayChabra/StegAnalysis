import { createContext, useContext, useState } from 'react';
import useHistory from '../hooks/useHistory.js';

// Shared app state: the session history (so Header's drawer and every page see
// the same log) and the drawer's open/close state.
const AppContext = createContext(null);

export function AppProvider({ children }) {
  const history = useHistory();
  const [historyOpen, setHistoryOpen] = useState(false);

  const value = {
    history,
    historyOpen,
    openHistory: () => setHistoryOpen(true),
    closeHistory: () => setHistoryOpen(false),
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within <AppProvider>');
  return ctx;
}
