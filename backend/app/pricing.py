"""Versioned model-price estimates; no default prices are guessed."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os


def pricing_table() -> tuple[str, str, dict]:
    version = os.getenv("RAGLAB_PRICING_VERSION", "unconfigured")
    updated_at = os.getenv("RAGLAB_PRICING_UPDATED_AT", "")
    raw = os.getenv("RAGLAB_PRICING_JSON") or "{}"
    try:
        table = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError("RAGLAB_PRICING_JSON must contain valid JSON")
    if not isinstance(table, dict):
        raise RuntimeError("RAGLAB_PRICING_JSON must be an object keyed by provider:model")
    return version, updated_at, table


def estimate_cost(provider: str, model: str, usage: dict, embedding_tokens: int = 0) -> dict:
    version, updated_at, table = pricing_table()
    key = f"{provider}:{model}"
    rates = table.get(key)
    if not isinstance(rates, dict):
        return {"estimated_usd": None, "pricing_key": key, "pricing_version": version, "pricing_updated_at": updated_at or None, "status": "unpriced"}
    input_rate = float(rates.get("input_per_million_tokens", 0))
    output_rate = float(rates.get("output_per_million_tokens", 0))
    embedding_rate = float(rates.get("embedding_per_million_tokens", 0))
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total = (prompt_tokens * input_rate + completion_tokens * output_rate + int(embedding_tokens) * embedding_rate) / 1_000_000
    return {
        "estimated_usd": round(total, 10),
        "pricing_key": key,
        "pricing_version": version,
        "pricing_updated_at": updated_at or None,
        "rates_per_million_tokens": rates,
        "status": "estimated",
        "estimated_at": datetime.now(timezone.utc).isoformat(),
    }
