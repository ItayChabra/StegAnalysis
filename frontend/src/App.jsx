import { useState } from 'react';
import './index.css';
import useAnalysis from './hooks/useAnalysis.js';
import Header                from './components/Header/Header.jsx';
import DropZone              from './components/DropZone/DropZone.jsx';
import AnalyzingIndicator    from './components/AnalyzingIndicator/AnalyzingIndicator.jsx';
import Dashboard             from './components/Dashboard/Dashboard.jsx';
import ImageComparisonSlider from './components/ImageComparisonSlider/ImageComparisonSlider.jsx';
import ErrorBanner           from './components/ErrorBanner/ErrorBanner.jsx';

export default function App() {
  const { state, result, error, analyze, reset } = useAnalysis();
  const [selectedPatch, setSelectedPatch] = useState(null);

  // Reset selected patch whenever a new analysis starts
  function handleAnalyze(file) {
    setSelectedPatch(null);
    analyze(file);
  }

  return (
    <>
      <Header />
      <main style={{ maxWidth: '960px', margin: '0 auto', padding: '0 24px 48px' }}>

        <DropZone analyze={handleAnalyze} state={state} />

        {state === 'ANALYZING' && (
          <AnalyzingIndicator />
        )}

        {state === 'COMPLETE' && (
          <>
            <Dashboard
              result={result}
              selectedPatch={selectedPatch}
              onCellClick={setSelectedPatch}
            />
            <ImageComparisonSlider
              jobId={result?.job_id}
              windowRows={result?.window_rows}
              windowCols={result?.window_cols}
              selectedPatch={selectedPatch}
            />
          </>
        )}

        {state === 'ERROR' && (
          <ErrorBanner error={error} reset={reset} />
        )}

      </main>
    </>
  );
}
