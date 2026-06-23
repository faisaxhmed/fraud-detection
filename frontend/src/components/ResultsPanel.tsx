import type { PredictionResponse, ReviewResponse } from '../api';
import { ShapChart } from './ShapChart';
import { Tooltip } from './Tooltip';
import { AgentReview } from './AgentReview';
import styles from './ResultsPanel.module.css';

interface Props {
  result: PredictionResponse | null;
  error: string | null;
  pending: boolean;
  caseNumber: number;
  timestamp: string | null;
  reviewResult: ReviewResponse | null;
  reviewError: string | null;
  reviewPending: boolean;
}

export function ResultsPanel({
  result,
  error,
  pending,
  caseNumber,
  timestamp,
  reviewResult,
  reviewError,
  reviewPending,
}: Props) {
  const caseLabel = `CASE No. ${String(caseNumber).padStart(4, '0')}`;

  if (pending) {
    return (
      <div className={styles.card}>
        <div className={styles.loading}>Scoring transaction…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.card}>
        <div className={styles.caseMeta}>
          <span>{caseLabel}</span>
          <span>{timestamp}</span>
        </div>
        <div className={styles.errorBox}>
          <p className={styles.errorTitle}>Request failed</p>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className={styles.card}>
        <div className={styles.empty}>Submit a transaction to open a case file.</div>
      </div>
    );
  }

  const isFlagged = result.prediction === 'fraud';
  const direction = isFlagged ? 'flagged' : 'cleared';

  return (
    <div className={styles.card}>
      <div className={styles.caseMeta}>
        <span>{caseLabel}</span>
        <span>DISTANCE ~{result.distance_km.toFixed(0)} KM</span>
        <span>{timestamp}</span>
      </div>

      <div className={styles.stampWrap}>
        <div className={`${styles.stamp} ${styles[direction]}`}>
          {isFlagged ? 'FLAGGED\nREVIEW REQUIRED' : 'CLEARED'}
        </div>
      </div>
      <p className={styles.confidenceLine}>MODEL CONFIDENCE: {(result.confidence * 100).toFixed(1)}%</p>
      <p className={styles.confidenceCaption}>How sure the model is in this verdict, not a guarantee, a probability.</p>

      <h3 className={styles.sectionTitle}>
        <span className={styles.sectionTitleRow}>
          Why: What Drove This Verdict
          <Tooltip text="SHAP attribution: a standard explainability method that scores how much each factor influenced the model's decision." />
        </span>
      </h3>
      <p className={styles.shapExplainer}>
        Each bar shows how strongly that factor pushed this case toward FLAGGED or CLEARED.
        <br />
        Ordered from strongest to weakest influence.
      </p>
      <ShapChart shapValues={result.shap_values} />

      <AgentReview reviewResult={reviewResult} reviewError={reviewError} reviewPending={reviewPending} />
    </div>
  );
}
