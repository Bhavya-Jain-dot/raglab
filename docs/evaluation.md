# Evaluation design

Retrieval and generation metrics are computed independently and retain per-example evidence. `Precision@K`, `Recall@K`, and MRR use explicit relevant document IDs; missing labels yield null recall/MRR, not a manufactured score. Answer F1 is token-overlap F1 against the provided reference answer. Faithfulness and answer relevancy are labeled heuristic proxies based on token overlap. Latency is measured around retrieval plus generation; provider token counts are recorded when supplied; cost remains null.

These metrics are not RAGAS Context Precision/Recall or an LLM judge. Query-only data can still produce retrieval traces and proxy relevancy but will have no reference-based answer F1. Aggregates include the example count; failed runs do not report completed metrics. A production evaluation engine still needs RAGAS, calibrated semantic evaluators, confidence intervals, evaluator-failure accounting, and metric quality tests.
