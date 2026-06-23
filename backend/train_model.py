"""Reproduces the notebook's XGBoost + class-weights model and persists
the artifacts the API needs: the fitted model, scaler, and label encoders.

The notebook (fraud_detection.ipynb) never saved these to disk - it only
displayed results inline - so this script re-runs that training path and
writes the artifacts /predict depends on.

Deployed variant: class-weights, not SMOTE. In the notebook's own
comparison, class-weights traded precision for recall (0.93 vs 0.80) -
deliberately chosen here because in fraud detection a missed fraud case
costs far more than a false alarm a human reviewer can dismiss. See the
scale_pos_weight comment below and README.md for the full rationale.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

CATEGORICAL_COLS = ["merchant", "category", "gender", "job"]
FEATURE_ORDER = [
    "merchant", "category", "amt", "gender", "city_pop", "job",
    "trans_hour", "trans_dayofweek", "age", "distance",
]

EARTH_RADIUS_KM = 6371


def haversine_km(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Great-circle distance in km - the flat-earth lat/long diff the notebook used understates
    real distance and distorts it differently depending on latitude."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the notebook's preprocess(): engineer time/age/distance features, drop identifiers."""
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    df["distance"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    # Dataset artifact: most merchant names are literally prefixed "fraud_" regardless of
    # whether the transaction is fraudulent - it's not real signal, just noisy naming.
    df["merchant"] = df["merchant"].str.replace("fraud_", "", regex=False)

    drop_cols = [
        "Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last",
        "street", "city", "state", "zip", "dob", "trans_num", "unix_time",
        "lat", "long", "merch_lat", "merch_long",
    ]
    return df.drop(columns=drop_cols)


def build_city_population_lookup(raw_df: pd.DataFrame) -> dict[str, int]:
    """Maps "City|State" -> city_pop, extracted before city/state get dropped from features.

    The API takes city + state from the user and looks up city_pop here, rather than asking
    for a population figure directly - city/state is what a person actually knows.
    """
    pairs = raw_df[["city", "state", "city_pop"]].drop_duplicates(subset=["city", "state"])
    return {f"{row.city}|{row.state}": int(row.city_pop) for row in pairs.itertuples()}


def build_category_options(clean_df: pd.DataFrame) -> dict[str, list[str]]:
    """Deduplicated, sorted value lists for merchant/category/job, post fraud_-stripping."""
    return {col: sorted(clean_df[col].unique().tolist()) for col in ["merchant", "category", "job"]}


def build_city_location_lookup(raw_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Maps "City|State" -> average cardholder {lat, long}, for resolving distance at predict time."""
    grouped = raw_df.groupby(["city", "state"])[["lat", "long"]].mean()
    return {
        f"{city}|{state}": {"lat": float(row.lat), "long": float(row.long)}
        for (city, state), row in grouped.iterrows()
    }


def build_merchant_location_lookup(raw_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Maps stripped merchant name -> {lat, long} averaged across every occurrence of that
    merchant in training data (not first-seen) - more representative when the same merchant
    name recurs at slightly different coordinates across transactions."""
    stripped = raw_df["merchant"].str.replace("fraud_", "", regex=False)
    grouped = raw_df.assign(merchant=stripped).groupby("merchant")[["merch_lat", "merch_long"]].mean()
    return {
        merchant: {"lat": float(row.merch_lat), "long": float(row.merch_long)}
        for merchant, row in grouped.iterrows()
    }


def report_city_pop_fairness(model: XGBClassifier, explainer_X: np.ndarray, feature_order: list[str]) -> None:
    """Prints the average SHAP contribution of city_pop on a sample, and which direction it pushes."""
    import shap

    sample = explainer_X[np.random.RandomState(42).choice(len(explainer_X), size=min(1000, len(explainer_X)), replace=False)]
    shap_values = shap.TreeExplainer(model).shap_values(sample)
    city_pop_idx = feature_order.index("city_pop")
    avg_shap = float(np.mean(shap_values[:, city_pop_idx]))
    direction = "toward fraud" if avg_shap > 0 else "toward legitimate"
    print(f"city_pop fairness check: avg SHAP = {avg_shap:+.4f} ({direction}, n={len(sample)})")


def main() -> None:
    """Trains XGBoost with class weights and saves model/scaler/encoders to backend/models/."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(REPO_ROOT, "fraudTrain.csv"))

    city_population_lookup = build_city_population_lookup(train_df)
    with open(os.path.join(MODEL_DIR, "city_population_lookup.json"), "w") as f:
        json.dump(city_population_lookup, f)

    # Built from raw lat/long before preprocess() drops those columns - lets predict.py compute
    # distance itself instead of trusting a user-typed number that can't be verified.
    city_location_lookup = build_city_location_lookup(train_df)
    with open(os.path.join(MODEL_DIR, "city_location_lookup.json"), "w") as f:
        json.dump(city_location_lookup, f)

    merchant_location_lookup = build_merchant_location_lookup(train_df)
    with open(os.path.join(MODEL_DIR, "merchant_location_lookup.json"), "w") as f:
        json.dump(merchant_location_lookup, f)

    train_clean = preprocess(train_df)

    category_options = build_category_options(train_clean)
    with open(os.path.join(MODEL_DIR, "category_options.json"), "w") as f:
        json.dump(category_options, f)

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        train_clean[col] = le.fit_transform(train_clean[col])
        encoders[col] = le

    X = train_clean[FEATURE_ORDER]
    y = train_clean["is_fraud"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # scale_pos_weight = legit/fraud ratio in this training set, not a guessed constant -
    # it tells XGBoost a missed fraud case is exactly that many times costlier than a false
    # alarm. This is what pushes recall up at the expense of precision: compliance cares more
    # about catching fraud than about flag volume, and the resulting higher flag rate is exactly
    # why the downstream GenAI triage layer (step 4) exists - to make that volume reviewable by
    # a human instead of decorative.
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_scaled, y)

    joblib.dump(model, os.path.join(MODEL_DIR, "fraud_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    joblib.dump(FEATURE_ORDER, os.path.join(MODEL_DIR, "feature_order.pkl"))

    print(f"Saved model, scaler, encoders, and feature order to {MODEL_DIR}")

    test_df = pd.read_csv(os.path.join(REPO_ROOT, "fraudTest.csv"))
    test_clean = preprocess(test_df)
    for col in CATEGORICAL_COLS:
        known = {label: idx for idx, label in enumerate(encoders[col].classes_)}
        test_clean[col] = test_clean[col].map(known).fillna(-1).astype(int)
    X_test_scaled = scaler.transform(test_clean[FEATURE_ORDER])
    report_city_pop_fairness(model, X_test_scaled, FEATURE_ORDER)


if __name__ == "__main__":
    main()
