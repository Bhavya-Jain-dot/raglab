# RAGLab

RAGLab is a local-first workbench for building, running, comparing, and inspecting retrieval-augmented generation pipelines. It includes a Next.js dashboard, FastAPI API, PostgreSQL/pgvector persistence, Celery/Redis experiment jobs, S3-compatible RustFS object storage, Phoenix trace export, and MLflow experiment logging.

## Start locally

Requirements: Docker Desktop with the Linux engine running and Docker Compose v2.

1. Open PowerShell in this folder.
2. Copy `.env.example` to `.env` and set unique values for `SECRET_KEY`, database password, and `S3_SECRET_KEY`. For long-lived encrypted provider keys, set a stable `CREDENTIAL_ENCRYPTION_KEY` (a Fernet key).
3. Run `docker compose up --build` and wait for the API and worker to become healthy.
4. Open [http://localhost:3000](http://localhost:3000) and create an account. API docs are at [http://localhost:8000/docs](http://localhost:8000/docs).
5. Create a project or load the synthetic demo. In Settings, add provider keys; in Pipelines, configure embeddings, retrieval, reranking, generation, and optional RAGAS metrics.

The local extractive baseline needs no provider key. External calls are opt-in and incur provider charges. The synthetic demo is illustrative and does not claim benchmark results. The first local FastEmbed run may download model weights.

## Included capabilities

- Argon2 accounts, short-lived access tokens, rotating and revocable refresh sessions.
- Workspace roles and project authorization, project/dataset/document management, versioned dataset snapshots, CSV/JSON/JSONL import/export, and supported file extraction.
- Fixed, sentence, recursive, and semantic chunking; FastEmbed/OpenAI/Google/Ollama embeddings; pgvector search; lexical/vector/hybrid/MMR retrieval and optional FastEmbed cross-encoder reranking.
- Extractive baseline and LiteLLM-backed OpenAI, Anthropic, Google, and Ollama generation. Retrieved text is passed as untrusted context.
- Durable Celery experiment jobs with progress, cancellation requests, retries, immutable configuration snapshots, per-example results, deterministic proxy metrics, and optional RAGAS metrics.
- Evidence-based experiment comparison, objective selection and bootstrap intervals; versioned prices are configurable, and unknown rates remain unpriced.
- OpenTelemetry/Phoenix span export, MLflow logging, trace inspection, audit events, encrypted user provider keys, and process-local API rate limiting.
- Dashboard views for datasets, pipelines, experiments, comparison, provider settings, the RAG playground, and saved playground traces.

## Configuration notes

- `RAGLAB_PRICING_JSON` is a JSON object keyed by `provider:model`, with `input_per_million_tokens`, `output_per_million_tokens`, and optional `embedding_per_million_tokens` rates. Set a pricing version and update timestamp alongside it. No prices are assumed by default.
- RAGAS is disabled by default because it performs extra model calls. Enable it per pipeline and select an evaluator provider/model. Failed or unavailable scores are stored as null with an error status. Generation token cost excludes RAGAS judge calls and is marked accordingly.
- Phoenix, MLflow, and RustFS use their Compose services by default. RustFS provides local S3-compatible object storage on ports 9000 (API) and 9001 (console), using one local storage path for development. This setup does not provide drive-level redundancy. For hosted deployment, configure persistent PostgreSQL/pgvector and object storage, HTTPS, backups, secret rotation, and migrations before exposing the service.
- Malware scanning is not bundled; configure an upload scanning service before production use.

## Developer checks

From `backend/`, install `requirements.txt` and run `pytest`. From `frontend/`, run `npm ci` and `npm run build`. With Docker Desktop ready, `docker compose up --build` exercises the complete local stack. GitHub Actions runs the backend test suite and frontend production build.

See `DEVELOPMENT_STATUS.md` and `docs/` for the current phase, API, architecture, evaluation semantics, and experiment behavior.
