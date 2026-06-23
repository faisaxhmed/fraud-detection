import { FEATURE_LABELS } from '../options';
import styles from './ShapChart.module.css';

interface Props {
  shapValues: Record<string, number>;
}

/** Notes shown as a native title tooltip on specific feature labels, where the label alone is ambiguous. */
const FEATURE_NOTES: Record<string, string> = {
  distance: 'Calculated by the backend from known city and merchant locations, not entered by the user.',
};

/** Plain-language strength tag for a SHAP magnitude, kept alongside the raw number, not instead of it. */
function strengthTag(absValue: number, isFraudPush: boolean): string {
  const strength = absValue > 1.0 ? 'Strong' : absValue >= 0.3 ? 'Moderate' : 'Slight';
  return `${strength} push toward ${isFraudPush ? 'FLAGGED' : 'CLEARED'}`;
}

/** Custom diverging bar chart: bars grow from a center zero-line toward fraud (coral, right) or cleared (ink, left). */
export function ShapChart({ shapValues }: Props) {
  const entries = Object.entries(shapValues).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.0001);

  return (
    <div className={styles.chart}>
      {entries.map(([feature, value], i) => {
        const widthPct = (Math.abs(value) / maxAbs) * 50;
        const isFraudPush = value >= 0;
        const direction = isFraudPush ? 'fraud' : 'legit';
        return (
          <div className={styles.row} key={feature}>
            <span className={styles.labelCol}>
              <span className={styles.exhibitTag}>Exhibit {i + 1}</span>
              <span className={styles.featureName} title={FEATURE_NOTES[feature]}>
                {FEATURE_LABELS[feature] ?? feature.replace(/_/g, ' ')}
                {FEATURE_NOTES[feature] ? ' *' : ''}
              </span>
            </span>
            <div className={styles.track}>
              <div className={styles.zeroLine} />
              <div className={`${styles.bar} ${styles[direction]}`} style={{ width: `${widthPct}%` }} />
            </div>
            <span className={styles.valueCol}>
              <span className={`${styles.value} ${styles[direction]}`}>
                {value >= 0 ? '+' : ''}
                {value.toFixed(2)}
              </span>
              <span className={`${styles.strengthTag} ${styles[direction]}`}>
                {strengthTag(Math.abs(value), isFraudPush)}
              </span>
            </span>
          </div>
        );
      })}
      <div className={styles.legend}>
        <span>← pushes legitimate</span>
        <span>pushes fraud →</span>
      </div>
    </div>
  );
}
