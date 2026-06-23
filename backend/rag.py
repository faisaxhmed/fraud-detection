"""Retrieval layer: grounds a prediction's top SHAP factors in fraud-policy knowledge base text.

Embeddings are computed locally with a small sentence-transformer (no API key, no network
call at request time) and stored in a persistent on-disk ChromaDB collection, built once from
backend/knowledge/ and reused across requests.
"""
import os

import chromadb
from chromadb.utils import embedding_functions

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BACKEND_DIR, "knowledge")
CHROMA_DIR = os.path.join(BACKEND_DIR, "chroma_db")
COLLECTION_NAME = "fraud_policy"

# Readable phrases for each model feature, used to build a retrieval query from SHAP factors -
# the feature names themselves (e.g. "trans_hour") are not natural language and retrieve poorly.
FEATURE_DESCRIPTIONS = {
    "amount": "high transaction amount",
    "merchant": "merchant identity and reputation",
    "category": "unusual merchant category",
    "gender": "cardholder gender",
    "city_pop": "city population of cardholder's location",
    "job": "cardholder occupation",
    "trans_hour": "late-night transaction timing",
    "trans_dayofweek": "weekend vs weekday spending pattern",
    "age": "cardholder age",
    "distance": "large cardholder-merchant distance",
}

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
_client = chromadb.PersistentClient(path=CHROMA_DIR)


def _load_documents() -> tuple[list[str], list[str], list[dict]]:
    """Reads each knowledge file into (ids, texts, metadatas)."""
    ids, texts, metadatas = [], [], []
    for filename in sorted(os.listdir(KNOWLEDGE_DIR)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(KNOWLEDGE_DIR, filename)
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        ids.append(filename)
        texts.append(text)
        metadatas.append({"source": filename})
    return ids, texts, metadatas


def _get_collection():
    """Gets (or builds, on first call) the persistent policy collection."""
    collection = _client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn)
    if collection.count() == 0:
        ids, texts, metadatas = _load_documents()
        collection.add(ids=ids, documents=texts, metadatas=metadatas)
    return collection


_collection = _get_collection()


def retrieve_policy(shap_values: dict[str, float], top_n_factors: int = 3, n_results: int = 3) -> list[str]:
    """Builds a query from the top contributing SHAP factors (by absolute magnitude) and
    retrieves the most relevant policy documents - grounding retrieval in *why* the model
    flagged this specific transaction, rather than in the raw transaction fields."""
    top_factors = sorted(shap_values.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n_factors]
    query = ", ".join(FEATURE_DESCRIPTIONS.get(feature, feature) for feature, _ in top_factors)

    results = _collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]


def retrieve_policy_by_query(query: str, n_results: int = 3) -> list[str]:
    """Retrieves policy documents for an arbitrary free-text query, rather than one built from
    SHAP factors. Used by the agent loop (agent.py) when the policy snippets already attached
    to a prediction aren't enough to decide confidently and it wants to look something up
    directly - same underlying collection as retrieve_policy(), just a different query source."""
    results = _collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]
