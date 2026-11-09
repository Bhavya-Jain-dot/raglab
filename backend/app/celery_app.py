import os
from celery import Celery

redis_url=os.getenv("REDIS_URL","redis://localhost:6379/0")
celery_app=Celery("raglab",broker=redis_url,backend=redis_url,include=["app.tasks"])
celery_app.conf.update(task_track_started=True,task_serializer="json",result_serializer="json",accept_content=["json"],worker_prefetch_multiplier=1,task_acks_late=True,task_reject_on_worker_lost=True,worker_cancel_long_running_tasks_on_connection_loss=True)
