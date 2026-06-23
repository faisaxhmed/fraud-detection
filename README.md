# Explainable Fraud Detector

A fraud detection system that explains its own decisions, built to demonstrate applied ML and GenAI engineering together rather than as separate exercises.

**Live demo**: [link]

## Problem

Out of every 1,000 card transactions, fewer than 6 are fraud. A model that predicts "legitimate" for everything scores over 99% accuracy and catches none of it. This project treats fraud detection as what it actually is: a high-stakes imbalanced classification problem, then extends it with the layer most fraud demos skip entirely: explaining *why* a transaction was flagged, in language a non-technical reviewer can actually act on.

## Architecture

The system runs as four connected stages, each independently testable:

1. **Predict** — XGBoost scores the transaction and returns a SHAP attribution per feature.
2. **Retrieve** — the top SHAP factors are used to query a small fraud-policy knowledge base (ChromaDB, local HuggingFace embeddings), returning the policy text most relevant to *this specific verdict*, not generic boilerplate.
3. **Explain** — an LLM call turns the SHAP values and retrieved policy into a plain-English narrative, grounded only in that evidence and explicitly instructed not to invent facts.
4. **Decide** — an agent runs a tool-calling loop: it can look up the merchant's historical fraud rate and pull additional policy before recommending escalate or clear, logging every decision and its reasoning to an audit trail.

Prediction is fast and free; the explanation and decision stages call an LLM and are deliberately separated into their own endpoints (`/narrative`, `/review`) so the core model stays testable at zero cost.

## Model evaluation

Three imbalance-handling strategies were compared across two model families:

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression (class weights) | 0.05 | 0.75 | 0.10 | 0.8484 |
| Random Forest (SMOTE) | 0.75 | 0.75 | 0.75 | 0.9849 |
| XGBoost (SMOTE) | 0.70 | 0.80 | 0.75 | 0.9896 |
| XGBoost (class weights) | 0.34 | 0.93 | 0.49 | 0.9970 |

**XGBoost with class weights was deployed deliberately, not because it has the best F1.** In a compliance context, missing real fraud is costlier than a false alarm a human can quickly dismiss, so the system is tuned for recall (0.93) over precision (0.34). This tradeoff is also why the GenAI layer exists: a high-recall model generates more flagged transactions, many of them false positives, and the narrative/agent layer is what makes that volume reviewable instead of overwhelming.

SHAP confirmed transaction amount as the dominant fraud signal, followed by merchant category and transaction hour. Gender was confirmed to have negligible predictive value, the correct and expected outcome for a protected attribute.

## Data quality, found and fixed

Three real issues were identified and corrected during development, not assumed clean from the source:

- **Merchant name artifact**: most merchant names in the source dataset carry a "fraud_" prefix regardless of actual fraud status, a documented quirk of this Sparkov-generated dataset. Stripped before training.
- **Distance miscalculation**: the original distance feature was a raw difference between latitude and longitude coordinates, not a real-world distance. Replaced with a proper haversine calculation in kilometers, and changed from a user-typed field to a value computed server-side from known merchant and cardholder locations, since asking a user to know the real distance between two addresses isn't a reasonable input to collect.
- **Fairness check**: city population (a demographic proxy) was checked the same way gender was, by reviewing its average SHAP attribution. The effect was modest and consistently noted in the AI's reasoning when relevant, rather than removed outright.

## Engineering practices

- Strict request validation on every endpoint (bounded fields, no unexpected input)
- Separate, tighter rate limits on the two LLM-backed endpoints, since each call has real cost
- Retrieved policy text and tool outputs are treated as data, never as instructions, an explicit prompt-injection defense
- The agent's tool loop has a hard 5-iteration cap; if no decision is reached, it defaults to escalate rather than failing silently
- Every agent decision and its reasoning is logged to a SQLite audit trail

## Tech stack

- **Model**: XGBoost, scikit-learn, SHAP, imbalanced-learn
- **Backend**: FastAPI, Python
- **RAG**: ChromaDB, HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **AI reasoning**: Claude API
- **Frontend**: React, TypeScript, CSS Modules
- **Audit log**: SQLite

## Dataset

Trained on the [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) on Kaggle, a public synthetic dataset of roughly 1.3 million transactions (0.58% fraud rate), generated using the Sparkov simulation tool. No real financial data is used. City, merchant, and category options in the live demo are limited to what exists in this dataset, roughly 1,000 simulated cardholders, not real-world coverage.

## A note on the AI features

The compliance narrative and agent review run on a personal, prepaid API budget. If that budget is exhausted, those two features may temporarily stop responding. The core fraud prediction and SHAP explanation don't depend on the AI layer and will keep working regardless.

## What this isn't

This is a portfolio project demonstrating how ML and GenAI engineering connect, not a production fraud system. It has no connection to real transactions, no regulatory approval, and is not certified for real-world use. See the in-app About page for a plainer explanation of what this demonstrates.

## Author

Built by Faisa Ahmed, 2026.
[LinkedIn] · [GitHub]
