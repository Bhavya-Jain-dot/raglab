"""Deterministic evaluators. Heuristic scores are explicitly named as proxies."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import re


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _f1(predicted: set[str], reference: set[str]) -> float:
    if not predicted and not reference:
        return 1.0
    common = len(predicted & reference)
    precision = common / max(1, len(predicted))
    recall = common / max(1, len(reference))
    return 2 * precision * recall / max(1e-12, precision + recall)


def _metric(score: float | None, method: str, explanation: str, *, low: float = 0.0, high: float = 1.0) -> dict:
    return {
        "score": round(float(score), 6) if score is not None else None,
        "range": [low, high],
        "status": "computed" if score is not None else "not_computed",
        "method": method,
        "explanation": explanation,
        "evaluator_version": "raglab-deterministic-v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_result(question: str, answer: str, retrieved: list[dict], reference_answer: str | None, relevant_document_ids: list | None, reference_contexts: list[str] | None = None, *, latency: dict | None = None, token_usage: dict | None = None, query_embedding: list[float] | None = None, answer_embedding: list[float] | None = None) -> tuple[dict[str, float | None], dict[str, dict]]:
    relevant = {str(item) for item in (relevant_document_ids or [])}
    retrieved_ids = [str(row.get("document_id")) for row in retrieved]
    hits = [doc_id in relevant for doc_id in retrieved_ids] if relevant else []
    first_rank = next((index for index, is_relevant in enumerate(hits, start=1) if is_relevant), None)

    if relevant:
        precision_k = sum(hits) / len(hits) if hits else 0.0
        found = len(set(retrieved_ids) & relevant)
        recall_k = found / len(relevant)
        mrr = 1 / first_rank if first_rank else 0.0
        dcg = sum((1 / math.log2(index + 1)) for index, is_relevant in enumerate(hits, start=1) if is_relevant)
        ideal = sum(1 / math.log2(index + 1) for index in range(1, min(len(relevant), len(hits)) + 1))
        ndcg = dcg / ideal if ideal else 0.0
        context_precision = sum((sum(hits[:index]) / index) for index, hit in enumerate(hits, start=1) if hit) / max(1, sum(hits))
    else:
        precision_k = recall_k = mrr = ndcg = context_precision = None

    context_text = "\n".join(str(row.get("text", "")) for row in retrieved)
    reference_contexts = reference_contexts or []
    if reference_contexts:
        required_terms = set().union(*(_tokens(context) for context in reference_contexts))
        context_recall = len(required_terms & _tokens(context_text)) / max(1, len(required_terms))
    elif relevant:
        context_recall = recall_k
    else:
        context_recall = None

    if reference_answer:
        answer_f1 = _f1(_tokens(answer), _tokens(reference_answer))
        exact_match = float(sorted(_tokens(answer)) == sorted(_tokens(reference_answer)))
        answer_correctness = answer_f1
    else:
        answer_f1 = exact_match = answer_correctness = None

    q_terms = _tokens(question)
    answer_terms = _tokens(answer)
    context_terms = _tokens(context_text)
    answer_relevancy_proxy = len(q_terms & answer_terms) / max(1, len(q_terms)) if answer else None
    faithfulness_proxy = len(answer_terms & context_terms) / max(1, len(answer_terms)) if answer and context_text else None
    unsupported_token_ratio = 1 - faithfulness_proxy if faithfulness_proxy is not None else None

    metrics: dict[str, float | None] = {
        "context_precision": context_precision,
        "context_recall": context_recall,
        "precision_at_k": precision_k,
        "recall_at_k": recall_k,
        "mrr": mrr,
        "ndcg_at_k": ndcg,
        "answer_f1": answer_f1,
        "answer_correctness_token_f1": answer_correctness,
        "answer_exact_match": exact_match,
        "faithfulness_overlap_proxy": faithfulness_proxy,
        "answer_relevancy_overlap_proxy": answer_relevancy_proxy,
        "unsupported_token_ratio_proxy": unsupported_token_ratio,
    }
    details = {
        "context_precision": _metric(context_precision, "binary relevance average precision", "Ranks relevant chunks ahead of irrelevant chunks using expected document IDs."),
        "context_recall": _metric(context_recall, "reference-context token coverage" if reference_contexts else "document-id recall", "Fraction of labeled relevant documents or reference-context tokens retrieved."),
        "precision_at_k": _metric(precision_k, "expected document IDs", "Relevant retrieved chunks divided by retrieved chunks."),
        "recall_at_k": _metric(recall_k, "expected document IDs", "Unique relevant document IDs retrieved divided by all labeled relevant document IDs."),
        "mrr": _metric(mrr, "expected document IDs", "Reciprocal rank of the first relevant retrieved chunk."),
        "ndcg_at_k": _metric(ndcg, "binary relevance DCG", "Normalized discounted cumulative gain using binary document relevance."),
        "answer_f1": _metric(answer_f1, "token overlap F1", "Lexical overlap with the reference answer; not a semantic judge."),
        "answer_correctness_token_f1": _metric(answer_correctness, "token overlap F1", "Deterministic reference-answer overlap signal."),
        "answer_exact_match": _metric(exact_match, "normalized exact match", "Whether normalized answer and reference tokens match exactly."),
        "faithfulness_overlap_proxy": _metric(faithfulness_proxy, "token support proxy", "Share of answer terms found in retrieved context; this is not a semantic faithfulness judgment."),
        "answer_relevancy_overlap_proxy": _metric(answer_relevancy_proxy, "question-answer token overlap", "Share of question terms present in the answer; this is not an LLM relevancy judgment."),
        "unsupported_token_ratio_proxy": _metric(unsupported_token_ratio, "token support proxy", "Share of answer terms not found in retrieved context; inspect the answer before treating as a hallucination."),
    }
    return metrics, details
