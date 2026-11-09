"""Provider-neutral embeddings with explicit models and lazy local runtimes."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import httpx


LOCAL_MODEL_ALIASES = {
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2": "intfloat/e5-base-v2",
}


def _as_vectors(value: Any) -> list[list[float]]:
    vectors = value.tolist() if hasattr(value, "tolist") else value
    return [[float(component) for component in vector] for vector in vectors]


@lru_cache(maxsize=4)
def _local_model(model: str):
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("Local embeddings require the fastembed package; rebuild the backend image.") from exc
    try:
        return TextEmbedding(model_name=LOCAL_MODEL_ALIASES.get(model, model))
    except Exception as exc:
        raise RuntimeError(f"Could not load local embedding model '{model}': {exc}") from exc


def embed_texts(texts: list[str], config: dict, api_key: str | None = None) -> list[list[float]]:
    """Embed a batch. Empty/lexical configurations intentionally return no vectors."""
    if not texts:
        return []
    provider = str(config.get("embedding_provider", "lexical")).lower()
    model = str(config.get("embedding_model") or "").strip()
    if provider in {"", "none", "lexical"}:
        return [[] for _ in texts]
    if not model:
        defaults = {"local": "sentence-transformers/all-MiniLM-L6-v2", "openai": "text-embedding-3-small", "google": "gemini-embedding-001", "ollama": "nomic-embed-text"}
        model = defaults.get(provider, "")
    if provider == "local":
        vectors = _local_model(model).embed(texts)
        return _as_vectors(list(vectors))

    timeout = httpx.Timeout(45.0, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        if provider == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("Add an OpenAI key in Settings or configure OPENAI_API_KEY on the backend.")
            base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            response = client.post(base + "/embeddings", headers={"Authorization": f"Bearer {key}"}, json={"model": model, "input": texts})
            response.raise_for_status()
            rows = sorted(response.json().get("data", []), key=lambda row: row.get("index", 0))
            return _as_vectors([row["embedding"] for row in rows])
        if provider == "google":
            key = api_key or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("Add a Google key in Settings or configure GOOGLE_API_KEY on the backend.")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
            requests = [{"model": f"models/{model}", "content": {"parts": [{"text": text}]}} for text in texts]
            response = client.post(url, params={"key": key}, json={"requests": requests})
            response.raise_for_status()
            return _as_vectors([row["values"] for row in response.json().get("embeddings", [])])
        if provider == "ollama":
            base = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")
            response = client.post(base + "/api/embed", json={"model": model, "input": texts})
            response.raise_for_status()
            return _as_vectors(response.json().get("embeddings", []))
    raise ValueError(f"Unsupported embedding provider: {provider}")


def embedding_key(config: dict) -> str:
    provider = str(config.get("embedding_provider", "lexical")).lower()
    model = config.get("embedding_model") or ("sentence-transformers/all-MiniLM-L6-v2" if provider == "local" else "")
    return f"{provider}:{model}" if provider not in {"", "none", "lexical"} else "lexical"
