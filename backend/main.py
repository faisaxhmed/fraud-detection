"""FastAPI app: middleware, rate limiting, and route registration."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent import run_agent_review
from audit_log import log_decision
from config import settings
from narrative import generate_narrative
from predict import predict_transaction
from schemas import (
    NarrativeRequest,
    NarrativeResponse,
    PredictionResponse,
    ReviewResponse,
    TransactionRequest,
)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Fraud Detection API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit(settings.rate_limit)
def predict(request: Request, transaction: TransactionRequest) -> PredictionResponse:
    """Scores a transaction for fraud and returns the verdict, confidence, and SHAP attributions."""
    return predict_transaction(transaction)


@app.post("/narrative", response_model=NarrativeResponse)
@limiter.limit(settings.narrative_rate_limit)
def narrative(request: Request, transaction: NarrativeRequest) -> NarrativeResponse:
    """Turns a prediction's SHAP values and retrieved policy text into a short compliance
    narrative via the Claude API. Rate-limited separately from /predict since each call has
    real, per-request LLM cost."""
    return generate_narrative(transaction)


@app.post("/review", response_model=ReviewResponse)
@limiter.limit(settings.review_rate_limit)
def review(request: Request, transaction: TransactionRequest) -> ReviewResponse:
    """Runs predict -> RAG retrieval -> the tool-calling agent loop (agent.py) and returns the
    agent's final escalate/clear decision, its reasoning, and a full tool-call trace. The
    tightest rate limit of the three LLM-backed endpoints, since one request here can trigger
    up to agent_max_iterations Claude calls, not just one."""
    prediction = predict_transaction(transaction)
    decision, reasoning, trace = run_agent_review(
        prediction.prediction,
        prediction.confidence,
        transaction.merchant,
        transaction.category,
        prediction.shap_values,
        prediction.relevant_policy,
    )

    top_factors = sorted(prediction.shap_values.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    log_decision(
        transaction_summary=transaction.model_dump(),
        shap_top_factors=top_factors,
        tools_called=[record.model_dump() for record in trace],
        decision=decision,
        reasoning=reasoning,
    )

    return ReviewResponse(
        prediction=prediction.prediction,
        confidence=prediction.confidence,
        decision=decision,
        reasoning=reasoning,
        trace=trace,
    )
