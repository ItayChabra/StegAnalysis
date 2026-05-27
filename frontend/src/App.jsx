import { Routes, Route, Navigate } from 'react-router-dom';
import './index.css';
import Header from './components/Header/Header.jsx';
import HistoryDrawer from './components/History/HistoryDrawer.jsx';
import AnalyzePage from './pages/AnalyzePage.jsx';
import EmbedPage from './pages/EmbedPage.jsx';
import ExtractPage from './pages/ExtractPage.jsx';
import LearnPage from './pages/LearnPage.jsx';

export default function App() {
  return (
    <>
      <Header />
      <HistoryDrawer />
      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/analyze" replace />} />
          <Route path="/analyze" element={<AnalyzePage />} />
          <Route path="/embed" element={<EmbedPage />} />
          <Route path="/extract" element={<ExtractPage />} />
          <Route path="/learn" element={<LearnPage />} />
          <Route path="*" element={<Navigate to="/analyze" replace />} />
        </Routes>
      </main>
    </>
  );
}
