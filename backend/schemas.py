"""Pydantic request/response models for the /predict and /narrative endpoints."""
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class TransactionRequest(BaseModel):
    """A single transaction, matching the features the model was trained on."""

    model_config = ConfigDict(extra="forbid")

    amount: float = Field(gt=0, le=1_000_000, description="Transaction amount in USD")
    merchant: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=100)
    gender: Literal["M", "F"]
    city: str = Field(min_length=1, max_length=100, description="Cardholder city")
    state: str = Field(min_length=1, max_length=2, description="Cardholder state, 2-letter code")
    job: str = Field(min_length=1, max_length=200)
    trans_hour: int = Field(ge=0, le=23, description="Hour of transaction, 24h clock")
    trans_dayofweek: int = Field(ge=0, le=6, description="0=Monday ... 6=Sunday")
    age: int = Field(ge=0, le=120, description="Cardholder age in years")


class PredictionResponse(BaseModel):
    """Model output: a fraud verdict, its confidence, per-feature SHAP attributions, and the
    distance the backend computed from known city/merchant locations (not user-supplied)."""

    prediction: Literal["fraud", "legitimate"]
    confidence: float
    shap_values: dict[str, float]
    distance_km: float
    relevant_policy: list[str]


_FeatureName = Annotated[str, StringConstraints(min_length=1, max_length=100)]
_PolicySnippet = Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class NarrativeRequest(BaseModel):
    """The subset of a /predict response needed to write a compliance narrative.

    Mirrors /predict's own output shape rather than accepting a raw transaction, so this
    endpoint never re-derives or re-validates the prediction itself.
    """

    model_config = ConfigDict(extra="forbid")

    prediction: Literal["fraud", "legitimate"]
    confidence: float = Field(ge=0, le=1)
    shap_values: dict[_FeatureName, float] = Field(min_length=1, max_length=20)
    relevant_policy: list[_PolicySnippet] = Field(min_length=1, max_length=5)


class NarrativeResponse(BaseModel):
    """Structured output from the Claude API: a short narrative and a one-line summary."""

    narrative: str
    summary_one_line: str


class ToolCallRecord(BaseModel):
    """One tool call the agent made during a /review run, and what it got back - the audit
    trace, so a reviewer can see *how* the agent reached its decision, not just the verdict."""

    tool: str
    arguments: dict
    result: Any


class ReviewResponse(BaseModel):
    """Output of the agent loop: the underlying prediction, the agent's final decision and
    reasoning, and a full trace of every tool call made along the way."""

    prediction: Literal["fraud", "legitimate"]
    confidence: float
    decision: Literal["escalate", "clear"]
    reasoning: str
    trace: list[ToolCallRecord]
