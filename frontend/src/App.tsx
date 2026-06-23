import { useState } from 'react';
import { TransactionForm } from './components/TransactionForm';
import { ResultsPanel } from './components/ResultsPanel';
import { Nav, type View } from './components/Nav';
import { Footer } from './components/Footer';
import { AboutView } from './components/AboutView';
import { CaseStudyView } from './components/CaseStudyView';
import {
  predict,
  review,
  ApiError,
  type PredictionResponse,
  type ReviewResponse,
  type TransactionRequest,
} from './api';
import styles from './App.module.css';

function App() {
  const [view, setView] = useState<View>('product');

  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [caseNumber, setCaseNumber] = useState(0);
  const [timestamp, setTimestamp] = useState<string | null>(null);

  const [reviewResult, setReviewResult] = useState<ReviewResponse | null>(null);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [reviewPending, setReviewPending] = useState(false);

  // Remounts TransactionForm back to its defaults on Clear, without touching caseNumber - the
  // next real submission should still increment from where it left off, not restart at 1.
  const [formKey, setFormKey] = useState(0);

  async function handleSubmit(transaction: TransactionRequest) {
    setPending(true);
    setError(null);
    setReviewResult(null);
    setReviewError(null);
    setReviewPending(false);
    setCaseNumber((n) => n + 1);
    setTimestamp(new Date().toLocaleString());
    try {
      const response = await predict(transaction);
      setResult(response);
    } catch (err) {
      setResult(null);
      setError(err instanceof ApiError ? err.message : 'Could not reach the backend.');
      setPending(false);
      return;
    }
    setPending(false);

    setReviewPending(true);
    try {
      const reviewResponse = await review(transaction);
      setReviewResult(reviewResponse);
    } catch (err) {
      setReviewError(err instanceof ApiError ? err.message : 'Could not reach the backend.');
    } finally {
      setReviewPending(false);
    }
  }

  function handleClear() {
    setResult(null);
    setError(null);
    setPending(false);
    setReviewResult(null);
    setReviewError(null);
    setReviewPending(false);
    setFormKey((k) => k + 1);
  }

  return (
    <div className={styles.page}>
      <Nav view={view} onNavigate={setView} />

      {view === 'product' && (
        <>
          <header className={styles.header}>
            <div>
              <h1 className={styles.title}>Explainable Fraud Detector</h1>
              <p className={styles.subtitle}>
                Every transaction gets a case file. Every verdict comes with its evidence.
              </p>
            </div>
            <button type="button" className={styles.clearButton} onClick={handleClear}>
              Clear
            </button>
          </header>
          <main className={styles.layout}>
            <TransactionForm key={formKey} onSubmit={handleSubmit} pending={pending} />
            <ResultsPanel
              result={result}
              error={error}
              pending={pending}
              caseNumber={caseNumber}
              timestamp={timestamp}
              reviewResult={reviewResult}
              reviewError={reviewError}
              reviewPending={reviewPending}
            />
          </main>
        </>
      )}

      {view === 'about' && <AboutView />}
      {view === 'caseStudy' && <CaseStudyView />}

      <Footer />
    </div>
  );
}

export default App;
