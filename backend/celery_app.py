import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery = Celery(
    "viralclips",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # One task at a time per worker process — prevents GPU/memory contention
    worker_prefetch_multiplier=1,
    # Re-queue tasks if a worker dies mid-flight
    task_acks_late=True,
    # Visibility timeout should exceed the longest task (Runway polling = ~10 min)
    broker_transport_options={"visibility_timeout": 7200},
)
