import re

from .celery_app import celery_app
from .database import SessionLocal
from .engine import run_experiment
from .models import Experiment


@celery_app.task(bind=True, max_retries=2, acks_late=True, reject_on_worker_lost=True)
def execute_experiment(self, experiment_id: int):
    try:
        run_experiment(experiment_id)
    except Exception as exc:
        db = SessionLocal()
        try:
            experiment = db.get(Experiment, experiment_id)
            if not experiment:
                raise
            safe_message = re.sub(r"(?i)([?&](?:key|api_key|token|access_token)=)[^&\s]+", r"\1[REDACTED]", str(exc))[:2000]
            if self.request.retries < self.max_retries and not experiment.cancel_requested:
                experiment.status = "queued"
                experiment.error = f"Retry {self.request.retries + 1}/{self.max_retries}: {safe_message}"
                db.commit()
                raise self.retry(exc=exc, countdown=min(2 ** (self.request.retries + 1), 30))
            experiment.status = "failed"
            experiment.error = safe_message
            db.commit()
        finally:
            db.close()
        raise
