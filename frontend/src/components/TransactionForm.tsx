import { useState, type FormEvent } from 'react';
import { validateTransaction, type TransactionRequest } from '../api';
import { categoryOptions, cityStateOptions, CATEGORY_LABELS, FEATURE_LABELS } from '../options';
import { Tooltip } from './Tooltip';
import styles from './TransactionForm.module.css';

const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

const DEFAULTS: TransactionRequest = {
  amount: 250.75,
  merchant: 'Rippin, Kub and Mann',
  category: 'grocery_pos',
  gender: 'F',
  city: 'Moravian Falls',
  state: 'NC',
  job: 'Psychologist, counselling',
  trans_hour: 23,
  trans_dayofweek: 5,
  age: 34,
};

interface Props {
  onSubmit: (transaction: TransactionRequest) => void;
  pending: boolean;
}

/** Transaction input card; validates against backend bounds client-side before a request is ever sent. */
export function TransactionForm({ onSubmit, pending }: Props) {
  const [draft, setDraft] = useState<TransactionRequest>(DEFAULTS);
  const [errors, setErrors] = useState<Record<string, string>>({});

  function update<K extends keyof TransactionRequest>(key: K, value: TransactionRequest[K]) {
    setDraft((prev) => ({ ...prev, [key]: value }));
  }

  function handleCityStateChange(key: string) {
    const [city, state] = key.split('|');
    setDraft((prev) => ({ ...prev, city, state }));
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const validationErrors = validateTransaction(draft);
    setErrors(validationErrors);
    if (Object.keys(validationErrors).length === 0) {
      onSubmit(draft);
    }
  }

  const inputClass = (field: string) => `${styles.input} ${errors[field] ? styles.invalid : ''}`;

  return (
    <form className={styles.card} onSubmit={handleSubmit}>
      <h2 className={styles.title}>Transaction</h2>
      <div className={styles.grid}>
        <Field label={FEATURE_LABELS.amount} tooltip="How much the transaction was for, in US dollars." error={errors.amount}>
          <input
            className={inputClass('amount')}
            type="number"
            step="0.01"
            value={draft.amount}
            onChange={(e) => update('amount', Number(e.target.value))}
          />
        </Field>

        <Field
          label={FEATURE_LABELS.merchant}
          tooltip="The business where the transaction happened. This demo can only recognize businesses it saw during training, which is why it's a dropdown instead of free typing."
          error={errors.merchant}
        >
          <select
            className={styles.select}
            value={draft.merchant}
            onChange={(e) => update('merchant', e.target.value)}
          >
            {categoryOptions.merchant.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </Field>

        <Field
          label={FEATURE_LABELS.category}
          tooltip="What kind of purchase this was, like groceries, travel, or entertainment."
          error={errors.category}
        >
          <select
            className={styles.select}
            value={draft.category}
            onChange={(e) => update('category', e.target.value)}
          >
            {categoryOptions.category.map((c) => (
              <option key={c} value={c}>
                {CATEGORY_LABELS[c] ?? c}
              </option>
            ))}
          </select>
        </Field>

        <Field
          label={FEATURE_LABELS.job}
          tooltip="The cardholder's occupation. Limited to job titles the model saw during training, same as Merchant."
          error={errors.job}
        >
          <select className={styles.select} value={draft.job} onChange={(e) => update('job', e.target.value)}>
            {categoryOptions.job.map((j) => (
              <option key={j} value={j}>
                {j}
              </option>
            ))}
          </select>
        </Field>

        <Field label={FEATURE_LABELS.gender} tooltip="The cardholder's gender, as recorded in this dataset." error={errors.gender}>
          <select
            className={styles.select}
            value={draft.gender}
            onChange={(e) => update('gender', e.target.value as 'M' | 'F')}
          >
            <option value="F">Female</option>
            <option value="M">Male</option>
          </select>
        </Field>

        <Field label={FEATURE_LABELS.age} tooltip="The cardholder's age in years." error={errors.age}>
          <input
            className={inputClass('age')}
            type="number"
            value={draft.age}
            onChange={(e) => update('age', Number(e.target.value))}
          />
        </Field>

        <Field
          label="City / State"
          tooltip="Where the cardholder's card is registered. This demo only includes about 1,000 simulated locations from training data, not real-world city coverage."
          error={errors.city}
          full
        >
          <select className={styles.select} value={`${draft.city}|${draft.state}`} onChange={(e) => handleCityStateChange(e.target.value)}>
            {cityStateOptions.map((opt) => (
              <option key={opt.key} value={opt.key}>
                {opt.label}
              </option>
            ))}
          </select>
        </Field>

        <Field
          label={FEATURE_LABELS.trans_hour}
          tooltip="What hour the transaction happened, on a 24-hour clock, so 11pm is 23."
          error={errors.trans_hour}
        >
          <input
            className={inputClass('trans_hour')}
            type="number"
            min={0}
            max={23}
            value={draft.trans_hour}
            onChange={(e) => update('trans_hour', Number(e.target.value))}
          />
        </Field>

        <Field
          label={FEATURE_LABELS.trans_dayofweek}
          tooltip="Which day of the week the transaction happened."
          error={errors.trans_dayofweek}
        >
          <select
            className={styles.select}
            value={draft.trans_dayofweek}
            onChange={(e) => update('trans_dayofweek', Number(e.target.value))}
          >
            {DAYS.map((day, i) => (
              <option key={day} value={i}>
                {day}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <button className={styles.submit} type="submit" disabled={pending}>
        {pending ? 'Scoring transaction…' : 'Run fraud check'}
      </button>
    </form>
  );
}

function Field({
  label,
  tooltip,
  error,
  full,
  children,
}: {
  label: string;
  tooltip?: string;
  error?: string;
  full?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className={`${styles.field} ${full || label.length > 20 ? styles.full : ''}`}>
      <span className={styles.labelRow}>
        <label className={styles.label}>{label}</label>
        {tooltip && <Tooltip text={tooltip} />}
      </span>
      {children}
      {error && <span className={styles.errorText}>{error}</span>}
    </div>
  );
}
