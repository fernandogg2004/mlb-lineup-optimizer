-- =============================================================================
-- Migration 002: Add confidence scorecard and defense report to gameday_predictions
-- =============================================================================
-- Idempotent: ADD COLUMN IF NOT EXISTS (PostgreSQL 9.6+).
-- Run once on the postgres-mlb container before the next DAG execution:
--
--   docker exec -i <postgres-mlb-container> psql -U $POSTGRES_USER -d $POSTGRES_DB \
--       < deploy/migrations/002_add_confidence_columns.sql
-- =============================================================================

ALTER TABLE gameday_predictions
    ADD COLUMN IF NOT EXISTS confidence_score   FLOAT,
    ADD COLUMN IF NOT EXISTS confidence_level   VARCHAR(10),
    ADD COLUMN IF NOT EXISTS scorecard_json     JSONB,
    ADD COLUMN IF NOT EXISTS defense_report_md  TEXT;

-- Partial index: fast lookup of games where the model is confident
CREATE INDEX IF NOT EXISTS idx_gdp_confidence_level
    ON gameday_predictions (confidence_level)
    WHERE confidence_level IS NOT NULL;

-- Descending index on predicted_at used by GET /v1/optimize/{game_pk}
CREATE INDEX IF NOT EXISTS idx_gdp_predicted_at_desc
    ON gameday_predictions (predicted_at DESC);

COMMENT ON COLUMN gameday_predictions.confidence_score IS
    'LineupScorecard.confidence_overall [0,1]. NULL for preliminary (fast_mode) predictions.';

COMMENT ON COLUMN gameday_predictions.confidence_level IS
    'Semaphore tier from LineupScorecard: HIGH (>=0.80) | MEDIUM (0.60-0.79) | LOW (<0.60). NULL for preliminary.';

COMMENT ON COLUMN gameday_predictions.scorecard_json IS
    'Full LineupScorecard serialized as JSONB: slot confidence, uncertainty_flags, top_drivers, etc.';

COMMENT ON COLUMN gameday_predictions.defense_report_md IS
    'Frozen Markdown content of the RAG defense report (CrisisDefenseReport.markdown_content). '
    'NULL when no crisis trigger fired. Stored at prediction time to guarantee determinism in the dugout.';
