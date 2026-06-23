/** Request/response types and client-side validation mirroring backend/schemas.py exactly. */

export interface TransactionRequest {
  amount: number;
  merchant: string;
  category: string;
  gender: 'M' | 'F';
  city: string;
  state: string;
  job: string;
  trans_hour: number;
  trans_dayofweek: number;
  age: number;
}

export interface PredictionResponse {
  prediction: 'fraud' | 'legitimate';
  confidence: number;
  shap_values: Record<string, number>;
  distance_km: number;
}

export interface ToolCallRecord {
  tool: string;
  arguments: Record<string, unknown>;
  result: unknown;
}

export interface ReviewResponse {
  prediction: 'fraud' | 'legitimate';
  confidence: number;
  decision: 'escalate' | 'clear';
  reasoning: string;
  trace: ToolCallRecord[];
}

export const BASE_URL = 'http://127.0.0.1:8000';
export const API_URL = `${BASE_URL}/predict`;
export const REVIEW_URL = `${BASE_URL}/review`;

/** Numeric bounds copied from backend/schemas.py — kept here so bad input never burns a rate-limited request. */
export const BOUNDS = {
  amount: { gt: 0, le: 1_000_000 },
  trans_hour: { ge: 0, le: 23 },
  trans_dayofweek: { ge: 0, le: 6 },
  age: { ge: 0, le: 120 },
} as const;

/** Validates a draft transaction against backend bounds; returns a field -> message map, empty if valid. */
export function validateTransaction(t: Partial<TransactionRequest>): Record<string, string> {
  const errors: Record<string, string> = {};

  if (t.amount === undefined || !(t.amount > BOUNDS.amount.gt)) {
    errors.amount = `Must be greater than ${BOUNDS.amount.gt}`;
  } else if (t.amount > BOUNDS.amount.le) {
    errors.amount = `Must be at most ${BOUNDS.amount.le.toLocaleString()}`;
  }

  if (!t.merchant?.trim()) errors.merchant = 'Required';
  if (!t.category?.trim()) errors.category = 'Required';
  if (!t.job?.trim()) errors.job = 'Required';
  if (t.gender !== 'M' && t.gender !== 'F') errors.gender = 'Required';
  if (!t.city?.trim() || !t.state?.trim()) errors.city = 'Required';

  if (t.trans_hour === undefined || t.trans_hour < BOUNDS.trans_hour.ge || t.trans_hour > BOUNDS.trans_hour.le) {
    errors.trans_hour = `Must be between ${BOUNDS.trans_hour.ge} and ${BOUNDS.trans_hour.le}`;
  }
  if (
    t.trans_dayofweek === undefined ||
    t.trans_dayofweek < BOUNDS.trans_dayofweek.ge ||
    t.trans_dayofweek > BOUNDS.trans_dayofweek.le
  ) {
    errors.trans_dayofweek = `Must be between ${BOUNDS.trans_dayofweek.ge} and ${BOUNDS.trans_dayofweek.le}`;
  }
  if (t.age === undefined || t.age < BOUNDS.age.ge || t.age > BOUNDS.age.le) {
    errors.age = `Must be between ${BOUNDS.age.ge} and ${BOUNDS.age.le}`;
  }

  return errors;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** Posts a transaction to /predict; surfaces the backend's actual 422/429 message rather than a generic failure. */
export async function predict(transaction: TransactionRequest): Promise<PredictionResponse> {
  const res = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(transaction),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const message =
      body?.error ??
      (Array.isArray(body?.detail)
        ? body.detail.map((d: { loc: string[]; msg: string }) => `${d.loc.at(-1)}: ${d.msg}`).join('; ')
        : body?.detail) ??
      `Request failed with status ${res.status}`;
    throw new ApiError(res.status, message);
  }

  return res.json();
}

/** Posts the same transaction to /review; runs the tool-calling agent loop and returns its
 * escalate/clear decision, reasoning, and tool-call trace. Slower and rate-limited tighter
 * than /predict, so callers should request it only after /predict has already returned. */
export async function review(transaction: TransactionRequest): Promise<ReviewResponse> {
  const res = await fetch(REVIEW_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(transaction),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const message =
      body?.error ??
      (Array.isArray(body?.detail)
        ? body.detail.map((d: { loc: string[]; msg: string }) => `${d.loc.at(-1)}: ${d.msg}`).join('; ')
        : body?.detail) ??
      `Request failed with status ${res.status}`;
    throw new ApiError(res.status, message);
  }

  return res.json();
}
