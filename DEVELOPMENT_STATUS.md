# RAGLab development status

Updated: 2026-11-08

## Phase status

### Phase 1 — Foundation

- [x] Next.js/TypeScript dashboard and FastAPI service.
- [x] PostgreSQL/SQLite ORM models, Alembic schema revisions, authentication, refresh rotation, and audit events.
- [x] Workspace membership roles and project-scoped authorization.
- [x] Compose configuration for frontend, API, PostgreSQL/pgvector, Redis/worker, RustFS S3-compatible storage, Phoenix, and MLflow.

### Phase 2 — RAG engine

- [x] PDF/TXT/DOCX/Markdown/CSV/JSON/JSONL text extraction, content deduplication, optional S3-compatible original storage, and document deletion.
- [x] Fixed, sentence, recursive, and semantic chunking; FastEmbed and hosted embedding providers; persisted chunk vectors and PostgreSQL pgvector search.
- [x] Lexical, vector, hybrid RRF, MMR, and optional FastEmbed cross-encoder reranking.
- [x] Extractive baseline and LiteLLM generation for OpenAI, Anthropic, Google, and Ollama; interactive RAG playground with saved traces.

### Phase 3 — Evaluation engine

- [x] Reference/query datasets, explicit splits, dataset versions/snapshots, example/document labels, and import/export.
- [x] Deterministic retrieval metrics and explicitly named overlap-proxy answer metrics.
- [x] Optional RAGAS collection metrics, gated per pipeline and stored with evaluator/model/status metadata. Missing or failed scores remain null.
- [ ] LLM answer relevancy/correctness and cost tracking for evaluator calls need additional end-to-end validation; RAGAS judge call costs are not included in generation cost totals.

### Phase 4 — Experimentation

- [x] Versioned pipeline configs, Celery async jobs, snapshot-based inputs, progress, cancellation requests, retry-as-new-run, per-example persistence, and comparison objectives/paired bootstrap intervals.
- [x] MLflow parameter and metric logging when configured.
- [ ] Repeated trials, full chart suite, and downloadable comparison reports are not implemented yet.

### Phase 5 — Observability

- [x] Phoenix-compatible OpenTelemetry span export for experiment and playground stages; trace IDs and trace details are persisted locally.
- [x] Stage latency, provider token usage, errors, retrieved chunks, and audit events are exposed in the API/dashboard.
- [ ] Phoenix ingestion and prompt/token detail behavior has not been verified against a running Compose stack. Sensitive prompt content is not exported in span attributes.

### Phase 6 — Optimization

- [x] Rule-based recommendations with evidence, confidence, expected trade-off, and next experiment; objective-aware comparison.
- [ ] Natural-language optimization assistant and prompt version/replay comparison are not implemented.

### Phase 7 — Product polish

- [x] Responsive local-first dashboard for projects, datasets, pipelines, experiments, provider settings, comparison, playground, and trace inspection; synthetic demo is labelled.
- [x] README, API, architecture, evaluation, and experiment docs.
- [ ] Full required route map, rich chart visualizations, document preview, profile/password recovery, and complete workspace switching are not implemented.

### Phase 8 — Testing and deployment

- [x] CI workflow, backend test cases for core logic/auth/workspace isolation, Dockerfiles, Compose services, environment template, persistent local volumes.
- [x] Backend Python sources compile with the bundled Python 3.12 runtime. Nine non-HTTP pytest cases pass (chunking, password hashing, workspace isolation, metric behavior, hybrid ranking, config validation, and pricing semantics).
- [ ] Two TestClient integration tests were not completed in this managed Windows sandbox: AnyIO cannot create its socketpair (`socket.accept` blocks), so the request never reaches FastAPI. These should be rerun on the user's machine or CI.
- [x] Frontend TypeScript check passed after the latest UI edits (`tsc --noEmit`).
- [ ] Latest Next.js production build could not run in this execution sandbox: SWC failed to canonicalize the project path with Windows access denied. The user had confirmed an earlier dashboard build succeeded before the latest UI additions.
- [ ] Docker Compose stack has not yet completed startup on the user's machine. The first run failed because the old `minio/minio:latest` image reference was unavailable. Compose now uses the official RustFS image; rerun `docker compose up --build` from PowerShell to pull it and verify containers and service connectivity.
- [ ] Playwright end-to-end tests, malware scanning integration, production hosting, HTTPS, backup/restore, and deployment secrets remain future work.

## Data integrity and limitations

- Synthetic demo records are not benchmark claims. Real experiment outputs are generated at run time.
- Retrieval labels without references are null, not zero. Heuristic answer signals are named proxies and should not be presented as semantic correctness or proven hallucination detection.
- Cost estimates are null until exact provider/model pricing is supplied. RAGAS evaluator calls are extra model calls and are not included in generation cost totals.
- Per-process rate limiting is suitable for a single local API process; production multi-worker deployments should use a shared Redis-backed limiter.
- Provider credentials are encrypted at rest with `CREDENTIAL_ENCRYPTION_KEY`, or a key derived from `SECRET_KEY`. A stable encryption secret is required to decrypt existing stored credentials after restarts.
- Malware scanning, public deployment, and production security review are not complete.
