"""Transparent lexical, vector, hybrid, and optional cross-encoder retrieval."""
from __future__ import annotations

import math
import re
from functools import lru_cache


def terms(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_l = math.sqrt(sum(a * a for a in left))
    norm_r = math.sqrt(sum(b * b for b in right))
    return dot / (norm_l * norm_r) if norm_l and norm_r else 0.0


def lexical_score(query: str, text: str) -> float:
    query_terms = set(terms(query))
    content_terms = set(terms(text))
    if not query_terms or not content_terms:
        return 0.0
    # Recall-weighted Jaccard is deterministic and avoids favoring tiny chunks.
    overlap = len(query_terms & content_terms)
    recall = overlap / len(query_terms)
    precision = overlap / len(content_terms)
    return 0.7 * recall + 0.3 * precision


def rank(candidates: list[dict], query: str, top_k: int, strategy: str = "similarity", query_embedding: list[float] | None = None, *, vector_order: list[int] | None = None, alpha: float = 0.5, mmr_lambda: float = 0.72) -> list[dict]:
    """Return top-k rows with comparable score components and stable ordering."""
    top_k = max(1, min(int(top_k), 100))
    scored = []
    for item in candidates:
        row = dict(item)
        lexical = lexical_score(query, str(row.get("text", "")))
        vector = max(0.0, min(1.0, (cosine(query_embedding, row.get("embedding")) + 1.0) / 2.0)) if query_embedding and row.get("embedding") else 0.0
        row["lexical_score"] = lexical
        row["similarity"] = vector if query_embedding else lexical
        scored.append(row)

    if strategy in {"vector", "semantic"} and query_embedding:
        if vector_order:
            order = {int(chunk_id): i for i, chunk_id in enumerate(vector_order)}
            scored.sort(key=lambda row: order.get(int(row.get("id", -1)), len(order) + 1))
        else:
            scored.sort(key=lambda row: (row["similarity"], row.get("document_id", 0), row.get("chunk_index", 0)), reverse=True)
    elif strategy in {"hybrid", "hybrid_mmr"} and query_embedding:
        # Reciprocal-rank fusion is robust to lexical and cosine score scales.
        lex = sorted(scored, key=lambda row: row["lexical_score"], reverse=True)
        if vector_order:
            server_order = {int(chunk_id): index for index, chunk_id in enumerate(vector_order)}
            vec = sorted(scored, key=lambda row: server_order.get(int(row.get("id", -1)), len(server_order) + 1))
        else:
            vec = sorted(scored, key=lambda row: cosine(query_embedding, row.get("embedding")), reverse=True)
        lex_rank = {id(row): i + 1 for i, row in enumerate(lex)}
        vec_rank = {int(row.get("id", id(row))): i + 1 for i, row in enumerate(vec)}
        for row in scored:
            row_id = int(row.get("id", id(row)))
            row["fusion_score"] = alpha / (60 + lex_rank[id(row)]) + (1 - alpha) / (60 + vec_rank.get(row_id, len(vec_rank) + 1))
        scored.sort(key=lambda row: row["fusion_score"], reverse=True)
        for row in scored:
            row["similarity"] = row["fusion_score"]
    else:
        scored.sort(key=lambda row: row["lexical_score"], reverse=True)

    if strategy in {"mmr", "hybrid_mmr"}:
        selected: list[dict] = []
        pool = list(scored)
        while pool and len(selected) < top_k:
            def utility(row: dict) -> float:
                base = row.get("similarity", row["lexical_score"])
                if query_embedding and row.get("embedding"):
                    redundancy = max((max(0.0, cosine(row["embedding"], prior.get("embedding"))) for prior in selected), default=0.0)
                else:
                    row_terms = set(terms(row.get("text", "")))
                    redundancy = max((len(row_terms & set(terms(p.get("text", "")))) / max(1, len(row_terms | set(terms(p.get("text", ""))))) for p in selected), default=0.0)
                return mmr_lambda * base - (1 - mmr_lambda) * redundancy
            best = max(pool, key=utility)
            best["mmr_score"] = utility(best)
            selected.append(best)
            pool.remove(best)
        scored = selected
    return scored[:top_k]


@lru_cache(maxsize=3)
def _cross_encoder(model_name: str):
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as exc:
        raise RuntimeError("Cross-encoder reranking requires fastembed; rebuild the backend image.") from exc
    return TextCrossEncoder(model_name=model_name)


def cross_encoder_rerank(query: str, candidates: list[dict], model_name: str, top_k: int) -> list[dict]:
    if not candidates:
        return []
    model = _cross_encoder(model_name)
    scores = list(model.rerank(query, [str(row.get("text", "")) for row in candidates]))
    reranked = []
    for row, score in zip(candidates, scores):
        value = dict(row)
        value["reranker_score"] = float(score)
        reranked.append(value)
    reranked.sort(key=lambda row: row["reranker_score"], reverse=True)
    return reranked[: max(1, int(top_k))]
