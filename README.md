# Credit Card Fraud Detection

## The Problem
Every time someone taps their card, a transaction happens. Millions occur every second worldwide. The vast majority are legitimate — but a tiny fraction are fraud. Someone stole a card, cloned it, or phished someone's details.

The bank needs to decide in milliseconds: approve or block?

Too aggressive — you block legitimate purchases and frustrate customers.
Too lenient — fraudsters drain accounts.

That split-second decision is a machine learning problem. And it's not a simple one.

---

## Why This Is Hard

### Challenge 1 — Class Imbalance
Out of every 1,000 transactions, maybe 1 or 2 are fraud. A model that just says "everything is legitimate" would be 99.8% accurate — and completely useless. It would catch zero fraud.

This is why accuracy is a meaningless metric here. Finding fraud is like finding 10 red balls hidden in 10,000 blue ones. A model that always says "blue" scores 99.9% but solves nothing.

### Challenge 2 — Anonymised Features
Real transaction data contains sensitive personal information — card numbers, names, locations. So in real-world datasets, features are anonymised using PCA (Principal Component Analysis). The dataset has features called V1, V2... V28. You don't know what they represent. You have to work with the mathematical relationships, not the raw meaning.

### Challenge 3 — Fraud Patterns Change
Fraudsters adapt. What worked last year won't work this year. Real fraud systems retrain constantly. This project acknowledges that limitation honestly — it's a snapshot model, not a live adaptive system.

---

## The Dataset
**Kaggle Credit Card Fraud Detection Dataset**
- 284,807 transactions from European cardholders (2013)
- 492 fraud cases — just 0.17% of all transactions
- Features V1–V28 are PCA-transformed for privacy
- Only `Time`, `Amount`, and `Class` are in original form

This is the industry-standard benchmark dataset for this problem.

---

## My Approach

### Step 1 — Understand the data
- Explore the class imbalance visually
- Analyse transaction amount and time distributions for fraud vs legitimate
- Understand what PCA-transformed features mean in practice

### Step 2 — Handle the imbalance (three approaches, compare all three)
- **SMOTE** — synthetically generate new fraud examples to balance the classes
- **Undersampling** — reduce legitimate transactions so classes are balanced
- **Class weights** — tell the model fraud matters more, penalise missing it heavily

The comparison between these three IS the interesting part of this project. Each has tradeoffs and the results will tell a story.

### Step 3 — Build and compare four models
| Model | Why |
|---|---|
| Logistic Regression | Baseline — simple, interpretable, fast |
| Random Forest | Handles imbalance well, robust to outliers |
| XGBoost | Industry standard for tabular ML, almost always wins |
| Isolation Forest | Specifically designed for anomaly detection — interesting alternative |

### Step 4 — Evaluate properly
Accuracy is useless here. The metrics that matter:
- **Precision** — of transactions I flagged as fraud, how many actually were? (Avoiding false accusations)
- **Recall** — of all actual fraud cases, how many did I catch? (Avoiding missed fraud)
- **F1 Score** — the balance between precision and recall
- **AUC-ROC** — overall model performance across all decision thresholds

### Step 5 — Explain the predictions (SHAP)
Use SHAP values to show which features drove each prediction. Not just "this is fraud" but "this is fraud because V14 and V4 are unusually high for this transaction amount."

Explainability is not optional in financial services — regulators require it.

---

## The Business Tradeoff (Important)
Precision vs recall is actually a business decision, not just a technical one:

A bank might prefer **high recall** — catch every fraud case even if you occasionally block a legitimate transaction and annoy a customer. The cost of missed fraud outweighs the cost of a frustrated customer.

A luxury retailer might prefer **high precision** — never block a legitimate high-value purchase, even if some fraud slips through. Losing a £5,000 sale to a false positive is worse than the occasional fraud.

This framing — showing you understand the business context of ML decisions — is what separates junior candidates from ones who get hired.

---

## What I'm Building
A clean dashboard where you can input transaction details and get:
- Real-time classification: legitimate or suspicious
- Confidence score
- SHAP explanation of which features drove the decision
- Visual breakdown of model performance (confusion matrix, ROC curve)

---

## Tech Stack
| Layer | Tech |
|---|---|
| ML models | Scikit-learn, XGBoost |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Data analysis | Pandas, NumPy, Matplotlib, Seaborn |
| Backend | FastAPI |
| Frontend | React |
| Hosting | Render + Vercel |

---

## Build Order (Step by Step)
1. Load and explore the dataset — understand the imbalance visually
2. Preprocess — scale `Amount` and `Time`, handle class imbalance
3. Build logistic regression baseline — get a working model first
4. Evaluate properly — confusion matrix, F1, AUC-ROC
5. Try SMOTE, undersampling, and class weights — compare results
6. Train Random Forest and XGBoost — compare against baseline
7. Try Isolation Forest as an anomaly detection alternative
8. Add SHAP explainability to the best model
9. Build FastAPI backend — wrap model in an API
10. Build React dashboard — input transaction, see prediction + explanation
11. Deploy and document findings honestly

---

## What I Want to Learn From This
- How to handle real-world class imbalance properly
- Why accuracy is misleading and which metrics actually matter
- How XGBoost works and why it dominates tabular ML
- How to use SHAP to make model decisions explainable
- The business context behind ML decisions in financial services

> I chose to compare four models rather than just picking XGBoost because the interesting finding is understanding *why* XGBoost wins on this type of data — and whether anomaly detection approaches like Isolation Forest perform differently to supervised classifiers. The comparison is the learning, not just the final model.

---

## Why This Matters for Fintech
Every fintech company — Revolut, Monzo, Starling, Wise, Stripe, PayPal — has a fraud team. This project shows I understand:
- The actual technical challenges of fraud detection (not just "I trained a model")
- How to work with imbalanced financial data
- Why explainability matters in regulated industries
- The business tradeoffs behind ML decisions

---

## Notes
- Build this properly but efficiently — 2 to 3 weeks, not 3 months
- The dataset is well understood — the value is in the analysis and explanation, not discovering something new
- Document every experiment: what I tried, what the numbers were, what I concluded
- Honest about limitations: this is a 2013 dataset, fraud patterns have changed significantly since
- Stretch goal: simulate a live transaction stream and score transactions in real time
- Stretch goal: build a simple alerting system — flag high-risk transactions above a threshold
