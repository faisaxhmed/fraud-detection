"""Loads the trained model/scaler/encoders once and runs predictions with SHAP explanations."""
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import HTTPException

from config import settings
from rag import retrieve_policy
from schemas import PredictionResponse, TransactionRequest

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), settings.model_dir)

# Maps internal training column names to the request schema's field names.
FEATURE_TO_FIELD = {"amt": "amount"}

_model = joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl"))
_scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
_feature_order = joblib.load(os.path.join(MODEL_DIR, "feature_order.pkl"))

with open(os.path.join(MODEL_DIR, "city_population_lookup.json")) as f:
    _city_population_lookup = json.load(f)

with open(os.path.join(MODEL_DIR, "city_location_lookup.json")) as f:
    _city_location_lookup = json.load(f)

with open(os.path.join(MODEL_DIR, "merchant_location_lookup.json")) as f:
    _merchant_location_lookup = json.load(f)

EARTH_RADIUS_KM = 6371


def _lookup_city_pop(city: str, state: str) -> int:
    """Looks up city_pop for a city/state pair; unknown combos are a clean 422, not a guess."""
    key = f"{city}|{state}"
    if key not in _city_population_lookup:
        raise HTTPException(status_code=422, detail=f"Unknown city/state combination: {city}, {state}")
    return _city_population_lookup[key]


def _resolve_distance_km(city: str, state: str, merchant: str) -> float:
    """Computes haversine distance from known city/merchant coordinates - distance is derived,
    never user-supplied, since a typed-in km figure can't be verified against anything."""
    city_key = f"{city}|{state}"
    if city_key not in _city_location_lookup:
        raise HTTPException(status_code=422, detail=f"Unknown city/state combination: {city}, {state}")
    if merchant not in _merchant_location_lookup:
        raise HTTPException(status_code=422, detail=f"Unknown merchant location: {merchant}")

    card = _city_location_lookup[city_key]
    merch = _merchant_location_lookup[merchant]
    lat1, lon1, lat2, lon2 = map(np.radians, (card["lat"], card["long"], merch["lat"], merch["long"]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a)))


# TreeExplainer is exact and fast for gradient-boosted trees, unlike the
# model-agnostic KernelExplainer the notebook would otherwise need.
_explainer = shap.TreeExplainer(_model)


def _encode_categorical(column: str, value: str) -> int:
    """Encodes a categorical value, falling back to -1 for categories unseen during training."""
    encoder = _encoders[column]
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return -1


def _build_feature_row(transaction: TransactionRequest, distance_km: float) -> pd.DataFrame:
    """Converts a validated request into the exact column order the model was trained on."""
    raw = {
        "merchant": _encode_categorical("merchant", transaction.merchant),
        "category": _encode_categorical("category", transaction.category),
        "amt": transaction.amount,
        "gender": _encode_categorical("gender", transaction.gender),
        "city_pop": _lookup_city_pop(transaction.city, transaction.state),
        "job": _encode_categorical("job", transaction.job),
        "trans_hour": transaction.trans_hour,
        "trans_dayofweek": transaction.trans_dayofweek,
        "age": transaction.age,
        "distance": distance_km,
    }
    return pd.DataFrame([raw], columns=_feature_order)


def predict_transaction(transaction: TransactionRequest) -> PredictionResponse:
    """Scores a transaction and returns the fraud verdict, confidence, SHAP attributions, and
    the backend-computed distance between cardholder and merchant."""
    distance_km = _resolve_distance_km(transaction.city, transaction.state, transaction.merchant)
    row = _build_feature_row(transaction, distance_km)
    scaled = _scaler.transform(row)

    # Model is trained with class weights tuned for recall (see train_model.py), so expect a
    # higher false-positive rate by design - that's the deliberate tradeoff, not a bug.
    fraud_probability = float(_model.predict_proba(scaled)[0, 1])
    prediction = "fraud" if fraud_probability >= 0.5 else "legitimate"
    confidence = fraud_probability if prediction == "fraud" else 1 - fraud_probability

    shap_row = _explainer.shap_values(scaled)[0]
    shap_values = {
        FEATURE_TO_FIELD.get(feature, feature): float(value)
        for feature, value in zip(_feature_order, shap_row)
    }

    relevant_policy = retrieve_policy(shap_values)

    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        shap_values=shap_values,
        distance_km=distance_km,
        relevant_policy=relevant_policy,
    )
