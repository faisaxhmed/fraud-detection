# Credit Card Fraud Detection

A machine learning pipeline for detecting fraudulent credit card transactions, built to demonstrate proper handling of class imbalance, model comparison, and explainable AI — evaluated on metrics that actually matter.

**Live demo:** Coming soon  
**Notebook:** [fraud_detection.ipynb](./fraud_detection.ipynb)

---

## Problem Statement

Out of every 1,000 card transactions, fewer than 6 are fraud. A model that predicts "legitimate" for everything scores 99%+ accuracy — and catches zero fraud. This project tackles that problem properly, treating fraud detection as what it actually is: a high-stakes imbalanced classification problem where the cost of a missed fraud and the cost of a false alarm are both real and different.

---

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

- 1,296,675 training transactions, 555,719 test transactions
- 0.58% fraud rate in training data
- Features include transaction amount, merchant, category, location, cardholder demographics and timestamps

---

## Approach

### Feature Engineering
Rather than encoding raw columns blindly, domain knowledge was used to engineer meaningful signals:

| Feature | Derived From | Reasoning |
|---|---|---|
| `trans_hour` | Transaction timestamp | Fraud peaks at late night hours |
| `trans_dayofweek` | Transaction timestamp | Weekend vs weekday spending patterns differ |
| `age` | Date of birth | Cardholder age is a behavioural signal |
| `distance` | Cardholder vs merchant coordinates | Transactions far from home address are suspicious |

### Handling Class Imbalance
Three strategies were compared:

| Strategy | How it works |
|---|---|
| Class weights | Penalises the model more for missing fraud, no data modification |
| Undersampling | Reduces legitimate transactions until classes balance |
| SMOTE | Generates synthetic fraud examples by interpolating between real ones |

### Models Trained
- Logistic Regression (class weights) — baseline
- XGBoost (SMOTE)
- XGBoost (class weights)
- Random Forest (SMOTE)

---

## Results

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression (class weights) | 0.05 | 0.75 | 0.10 | 0.8484 |
| Random Forest (SMOTE) | 0.75 | 0.75 | 0.75 | 0.9849 |
| XGBoost (SMOTE) | 0.70 | 0.80 | 0.75 | 0.9896 |
| XGBoost (class weights) | 0.34 | 0.93 | 0.49 | 0.9970 |

**Key finding:** XGBoost with SMOTE offers the best precision-recall balance (F1: 0.75). XGBoost with class weights achieves the highest recall (0.93) and AUC-ROC (0.9970) — catching more fraud but generating more false alarms.

The choice between these two is a business decision:
- **Prioritise catching all fraud** → XGBoost class weights (high recall)
- **Prioritise customer experience** → XGBoost SMOTE (fewer false alarms)

---

## Explainability (SHAP)

SHAP values were applied to the best-performing model to explain individual predictions. In financial services and public sector applications, automated decisions must be justifiable — "the model said so" is not acceptable.

**Top fraud signals identified:**
1. **Transaction amount** — the strongest predictor by far. Unusually high amounts are the primary red flag
2. **Merchant category** — certain categories (online, entertainment) carry higher fraud risk
3. **Transaction hour** — late night transactions are disproportionately fraudulent
4. **Day of week** — weekend vs weekday patterns differ between fraud and legitimate transactions
5. **Distance** — transactions far from the cardholder's registered location contribute a moderate signal

**Notable finding:** Gender showed minimal predictive value — the model learned to mostly ignore it, which is the correct and ethical outcome.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML models | Scikit-learn, XGBoost |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Data analysis | Pandas, NumPy, Matplotlib, Seaborn |

---

## How to Run

```bash
git clone https://github.com/yourusername/fraud-detection
cd fraud-detection
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and place `fraudTrain.csv` and `fraudTest.csv` in the project root.

---

## Future Work

The notebook is the foundation. The full product roadmap:

**FastAPI Backend**
Wrap the trained XGBoost model in a REST API with a `/predict` endpoint. Send a transaction, receive a fraud probability, confidence score, and SHAP explanation. This makes the model deployable in a real system.

**React Dashboard**
A clean frontend where a user inputs transaction details and sees:
- Fraud / Legitimate classification
- Confidence percentage
- SHAP feature importance chart for that specific transaction

**Deployment**
- Backend hosted on Render
- Frontend hosted on Vercel
- Live URL accessible to anyone

**Model Improvements**
- Time-based train/test split to better simulate real deployment conditions
- Isolation Forest as an unsupervised anomaly detection alternative
- Threshold tuning — adjusting the classification threshold to optimise for a specific business objective (e.g. maximise recall above 0.90 while keeping precision above 0.50)
- Live transaction stream simulation

**Limitations to address**
- Dataset is from 2019-2020 — fraud patterns have evolved significantly
- LabelEncoder applied separately to train and test — a production system would use a consistent saved encoder
- SMOTE generates synthetic data which may not reflect real fraud patterns perfectly

---

## Honest Limitations

This is a snapshot model trained on 2019-2020 data. Real fraud detection systems retrain continuously as fraud patterns evolve. The results here represent strong performance on a benchmark dataset — not a production-ready system.

---

*Built as part of a portfolio demonstrating applied ML for high-stakes classification problems.*