import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = os.environ["DATABASE_URL"]

# Supabase's pgBouncer pooler (port 6543) runs in transaction mode —
# it doesn't support prepared statements, so use NullPool (one connection
# per DB call, returned immediately). Ideal for stateless Cloud Run too.
from sqlalchemy.pool import NullPool  # noqa: E402

_supabase = "supabase" in DATABASE_URL
if _supabase:
    engine = create_engine(
        DATABASE_URL,
        poolclass=NullPool,
        connect_args={"sslmode": "require"},
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
