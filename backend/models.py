import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from database import Base


def _uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id                       = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    email                    = Column(String(255), unique=True, nullable=False)
    password_hash            = Column(String(255), nullable=False)
    tier                     = Column(String(20), nullable=False, default="free")
    email_verified           = Column(Boolean, nullable=False, default=False)
    email_verification_token = Column(String(100), nullable=True)
    created_at               = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at               = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    jobs   = relationship("Job", back_populates="user", cascade="all, delete-orphan")
    usages = relationship("UsageRecord", back_populates="user", cascade="all, delete-orphan")


class Job(Base):
    __tablename__ = "jobs"

    id                = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id           = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    status            = Column(String(50), nullable=False, default="pending")
    original_filename = Column(String(500), nullable=False)
    gcs_input_path    = Column(String(500))
    gcs_output_path   = Column(String(500))
    transcript        = Column(JSONB)
    clips_total       = Column(Integer, nullable=False, default=0)
    clips_done        = Column(Integer, nullable=False, default=0)
    error             = Column(Text)
    created_at        = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at        = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    user  = relationship("User", back_populates="jobs")
    clips = relationship("Clip", back_populates="job", cascade="all, delete-orphan")


class Clip(Base):
    __tablename__ = "clips"

    id           = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    job_id       = Column(UUID(as_uuid=False), ForeignKey("jobs.id"), nullable=False)
    start_time   = Column(Float, nullable=False)
    end_time     = Column(Float, nullable=False)
    reason       = Column(Text, nullable=False)
    hook_score   = Column(Integer)
    content_type = Column(String(50))
    status       = Column(String(50), nullable=False, default="pending")
    output_path  = Column(String(500))
    created_at   = Column(DateTime(timezone=True), default=datetime.utcnow)

    job = relationship("Job", back_populates="clips")


class UsageRecord(Base):
    __tablename__ = "usage_records"

    id                = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id           = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    billing_period    = Column(String(7), nullable=False)   # YYYY-MM
    videos_processed  = Column(Integer, nullable=False, default=0)
    minutes_processed = Column(Float, nullable=False, default=0.0)
    created_at        = Column(DateTime(timezone=True), default=datetime.utcnow)

    user = relationship("User", back_populates="usages")
