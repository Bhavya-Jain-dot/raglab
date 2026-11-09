"""Optional RAGAS collection metrics with explicit, per-metric failure reporting."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import partial
from typing import Any


def _detail(score: float | None, metric: str, model: str, error: str | None = None) -> dict[str, Any]:
    return {
        "score": score,
        "range": [0.0, 1.0],
        "status": "computed" if score is not None else ("error" if error else "not_computed"),
        "method": "RAGAS collection metric",
        "evaluator_model": model,
        "evaluator_version": "ragas-0.4+",
        "explanation": error or f"RAGAS {metric} evaluator signal; inspect the source answer and contexts.",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


async def _evaluate(question: str, answer: str, contexts: list[str], reference_answer: str | None, reference_contexts: list[str], model: str, api_key: str | None, api_base: str | None) -> tuple[dict[str, float | None], dict[str, dict]]:
    try:
        import litellm
        from ragas.llms import llm_factory
        from ragas.metrics.collections import ContextPrecision, ContextRecall, ContextUtilization, Faithfulness
    except ImportError as exc:
        raise RuntimeError("RAGAS evaluation requires the installed ragas and litellm packages.") from exc

    client = partial(litellm.completion, api_key=api_key, api_base=api_base)
    evaluator = llm_factory(model=model, provider="litellm", client=client, adapter="litellm")
    candidates = {
        "ragas_faithfulness": Faithfulness(llm=evaluator),
        "ragas_context_utilization": ContextUtilization(llm=evaluator),
    }
    if reference_answer:
        candidates["ragas_context_precision"] = ContextPrecision(llm=evaluator)
    if reference_contexts:
        candidates["ragas_context_recall"] = ContextRecall(llm=evaluator)

    scores: dict[str, float | None] = {}
    details: dict[str, dict] = {}
    for name, metric in candidates.items():
        try:
            common = {"user_input": question, "response": answer, "retrieved_contexts": contexts}
            if name == "ragas_context_precision":
                result = await metric.ascore(user_input=question, reference=reference_answer, retrieved_contexts=contexts)
            elif name == "ragas_context_recall":
                result = await metric.ascore(user_input=question, reference_contexts=reference_contexts, retrieved_contexts=contexts)
            else:
                result = await metric.ascore(**common)
            value = float(result.value) if result.value is not None else None
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"RAGAS returned out-of-range score {value}")
            scores[name] = value
            details[name] = _detail(value, name.removeprefix("ragas_"), model)
        except Exception as exc:
            # Keep failed/missing judgments null. Do not poison aggregate scores with zero.
            scores[name] = None
            details[name] = _detail(None, name.removeprefix("ragas_"), model, f"RAGAS evaluator failed: {type(exc).__name__}: {str(exc)[:300]}")
    for name in ("ragas_context_precision", "ragas_context_recall"):
        if name not in scores:
            scores[name] = None
            details[name] = _detail(None, name.removeprefix("ragas_"), model, "Required reference answer/context labels were not supplied; this metric was not computed.")
    return scores, details


def evaluate_ragas(question: str, answer: str, retrieved: list[dict], reference_answer: str | None, reference_contexts: list[str] | None, model: str, api_key: str | None = None, api_base: str | None = None) -> tuple[dict[str, float | None], dict[str, dict]]:
    """Synchronously score one result; this is opt-in because it makes model calls."""
    if not model.strip():
        raise ValueError("Set an explicit evaluation model to enable RAGAS.")
    contexts = [str(row.get("text", "")) for row in retrieved if str(row.get("text", "")).strip()]
    return asyncio.run(_evaluate(question, answer, contexts, reference_answer, reference_contexts or [], model, api_key, api_base))
