import { useEffect, useState } from 'react';
import type { ReviewResponse, ToolCallRecord } from '../api';
import { CATEGORY_LABELS } from '../options';
import { Tooltip } from './Tooltip';
import styles from './AgentReview.module.css';

interface Props {
  reviewResult: ReviewResponse | null;
  reviewError: string | null;
  reviewPending: boolean;
}

const LOADING_MESSAGES = [
  'Pulling case history…',
  'Cross-referencing policy…',
  'Weighing the evidence…',
  'Reaching a decision…',
];

function useRotatingMessage(active: boolean, messages: string[], intervalMs = 2200) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (!active) {
      setIndex(0);
      return;
    }
    const timer = setInterval(() => setIndex((i) => (i + 1) % messages.length), intervalMs);
    return () => clearInterval(timer);
  }, [active, messages, intervalMs]);

  return messages[index];
}

function ThinkingDots() {
  return (
    <span className={styles.dots} aria-hidden="true">
      <span className={styles.dot} />
      <span className={styles.dot} />
      <span className={styles.dot} />
    </span>
  );
}

interface PolicySummary {
  title: string;
  summary: string;
}

function describeToolCall(record: ToolCallRecord): { summary: string; detail: React.ReactNode } {
  if (record.tool === 'lookup_historical_pattern') {
    const merchant = String(record.arguments.merchant ?? 'this merchant');
    const rawCategory = String(record.arguments.category ?? 'this category');
    const category = CATEGORY_LABELS[rawCategory] ?? rawCategory;
    return {
      summary: `Checked how often ${merchant} in ${category} has been fraudulent before.`,
      detail: String(record.result ?? ''),
    };
  }

  if (record.tool === 'query_policy') {
    const query = String(record.arguments.query ?? 'this topic');
    const results = Array.isArray(record.result) ? (record.result as PolicySummary[]) : [];
    return {
      summary: `Searched company policy for guidance on "${query}".`,
      detail: (
        <div className={styles.policyResults}>
          {results.map((item, i) => (
            <div key={i} className={styles.policyResult}>
              <p className={styles.policyTitle}>{item.title}</p>
              <p className={styles.policyBody}>{item.summary}</p>
            </div>
          ))}
        </div>
      ),
    };
  }

  return { summary: record.tool, detail: null };
}

export function AgentReview({ reviewResult, reviewError, reviewPending }: Props) {
  const [traceOpen, setTraceOpen] = useState(false);
  const loadingMessage = useRotatingMessage(reviewPending, LOADING_MESSAGES);

  const visibleTrace = reviewResult?.trace.filter((record) => record.tool !== 'make_decision') ?? [];
  const reasoningParagraphs = reviewResult?.reasoning.split(/\n\s*\n/).filter((p) => p.trim().length > 0) ?? [];

  return (
    <div className={styles.section}>
      <h3 className={styles.sectionTitle}>
        <span className={styles.sectionTitleRow}>
          Agent Review
          <Tooltip text="The model predicts whether this transaction is fraud. The agent decides what a human should do about it, checking historical merchant/category data and policy text before answering." />
        </span>
      </h3>

      {reviewPending && (
        <div className={styles.loading}>
          <ThinkingDots />
          <span>{loadingMessage}</span>
        </div>
      )}

      {!reviewPending && reviewError && (
        <div className={styles.unavailable}>
          <p className={styles.unavailableTitle}>Agent review unavailable</p>
          <p>{reviewError}</p>
        </div>
      )}

      {!reviewPending && !reviewError && reviewResult && (
        <>
          <div className={styles.decisionWrap}>
            <div className={`${styles.decisionStamp} ${styles[reviewResult.decision]}`}>
              {reviewResult.decision === 'escalate' ? 'ESCALATE' : 'CLEAR'}
            </div>
          </div>
          <div className={styles.reasoning}>
            {reasoningParagraphs.map((paragraph, i) => (
              <p key={i}>{paragraph.trim()}</p>
            ))}
          </div>

          {visibleTrace.length > 0 && (
            <>
              <button
                type="button"
                className={styles.traceToggle}
                onClick={() => setTraceOpen((open) => !open)}
                aria-expanded={traceOpen}
              >
                {traceOpen ? 'Hide agent’s work' : 'Show agent’s work'}
              </button>

              {traceOpen && (
                <ul className={styles.trace}>
                  {visibleTrace.map((record, i) => {
                    const { summary, detail } = describeToolCall(record);
                    return (
                      <li key={i} className={styles.traceItem}>
                        <p className={styles.traceSummary}>{summary}</p>
                        {typeof detail === 'string' ? <p className={styles.traceDetail}>{detail}</p> : detail}
                      </li>
                    );
                  })}
                </ul>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
