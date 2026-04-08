-- ViralClips Postgres schema
-- Run once: psql $DATABASE_URL -f schema.sql

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Users ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    tier            VARCHAR(20)  NOT NULL DEFAULT 'free',   -- free | paid
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Jobs ─────────────────────────────────────────────────────────────────────
-- status flow: pending → downloading → transcribing → detecting →
--              reframing → zipping → complete | failed
CREATE TABLE IF NOT EXISTS jobs (
    id                  UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status              VARCHAR(50)  NOT NULL DEFAULT 'pending',
    original_filename   VARCHAR(500) NOT NULL,
    gcs_input_path      VARCHAR(500),        -- set after upload confirmed
    gcs_output_path     VARCHAR(500),        -- set when zip uploaded
    transcript          JSONB,               -- Whisper segments
    clips_total         INTEGER      NOT NULL DEFAULT 0,
    clips_done          INTEGER      NOT NULL DEFAULT 0,
    error               TEXT,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);

-- ── Clips ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS clips (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID        NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    start_time      FLOAT       NOT NULL,
    end_time        FLOAT       NOT NULL,
    reason          TEXT        NOT NULL,
    hook_score      INTEGER,
    content_type    VARCHAR(50),   -- single_face | dual_face | no_face
    status          VARCHAR(50) NOT NULL DEFAULT 'pending',
    output_path     VARCHAR(500),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clips_job_id ON clips(job_id);

-- ── Usage tracking ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS usage_records (
    id                  UUID       PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    billing_period      CHAR(7)    NOT NULL,   -- YYYY-MM
    videos_processed    INTEGER    NOT NULL DEFAULT 0,
    minutes_processed   FLOAT      NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, billing_period)
);

CREATE INDEX IF NOT EXISTS idx_usage_user_billing ON usage_records(user_id, billing_period);
