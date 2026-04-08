import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, default="pending")
    # pending → transcribing → detecting → reframing → complete | failed
    original_filename = Column(String, nullable=False)
    video_path = Column(String, nullable=False)
    transcript = Column(Text, nullable=True)   # JSON blob from Whisper
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    clips = relationship("Clip", back_populates="job", cascade="all, delete-orphan")


class Clip(Base):
    __tablename__ = "clips"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    start_time = Column(Float, nullable=False)   # seconds
    end_time = Column(Float, nullable=False)     # seconds
    reason = Column(Text, nullable=False)        # GPT-4o mini explanation
    runway_task_id = Column(String, nullable=True)
    output_path = Column(String, nullable=True)  # local path to 9:16 mp4
    status = Column(String, default="pending")
    # pending → processing → complete | failed
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job", back_populates="clips")
