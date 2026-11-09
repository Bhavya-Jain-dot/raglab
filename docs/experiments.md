# Experiments

An experiment pins dataset and pipeline versions/config, execution engine, and timestamps. It stores per-question retrieval results, answer, latency, token usage, proxy scores, and trace spans. Celery queues work through Redis and exposes queued/running/completed/failed/cancelled states. Repeated trials are separate experiments.

Current provider/model values are recorded in the pipeline configuration, but Git commit, artifact bundles, cost rates, and external MLflow tracking are not implemented. Provider outputs may vary; actual outputs are retained. Cancellation checks between dataset examples; a worker crash may still require operator cleanup.
