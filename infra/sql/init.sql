-- init.sql — Schema inicial de PostgreSQL para MLB AI
-- Se ejecuta automáticamente cuando postgres arranca por primera vez.

CREATE DATABASE mlflow;
\c mlb_predictions;

-- Tabla principal de predicciones (output del pipeline diario)
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    game_date       DATE          NOT NULL,
    game_pk         INTEGER       NOT NULL,
    team_abbr       VARCHAR(10)   NOT NULL,
    side            VARCHAR(10)   NOT NULL CHECK (side IN ('home','away')),
    batting_order   JSONB         NOT NULL,
    expected_runs   NUMERIC(5,2),
    model_version   VARCHAR(100),
    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   DEFAULT NOW(),

    CONSTRAINT uq_prediction UNIQUE (game_date, game_pk, team_abbr, side)
);

CREATE INDEX idx_predictions_game_date  ON predictions (game_date);
CREATE INDEX idx_predictions_team_abbr  ON predictions (team_abbr);
CREATE INDEX idx_predictions_model_ver  ON predictions (model_version);

-- Tabla de resultados shadow (modelo challenger)
CREATE TABLE IF NOT EXISTS shadow_predictions (
    id              BIGSERIAL PRIMARY KEY,
    game_date       DATE          NOT NULL,
    game_pk         INTEGER       NOT NULL,
    team_abbr       VARCHAR(10)   NOT NULL,
    side            VARCHAR(10)   NOT NULL,
    batting_order   JSONB         NOT NULL,
    expected_runs   NUMERIC(5,2),
    model_path      TEXT,
    latency_ms      NUMERIC(8,1),
    created_at      TIMESTAMPTZ   DEFAULT NOW(),

    CONSTRAINT uq_shadow UNIQUE (game_date, game_pk, team_abbr, side, model_path)
);

-- Tabla de auditoría de drift diario
CREATE TABLE IF NOT EXISTS drift_reports (
    id              BIGSERIAL PRIMARY KEY,
    report_date     DATE          NOT NULL UNIQUE,
    psi_total       NUMERIC(8,4),
    psi_per_class   JSONB,
    psi_features    JSONB,
    status          VARCHAR(20),  -- ok | warning | critical | no_data
    n_predictions   INTEGER,
    created_at      TIMESTAMPTZ   DEFAULT NOW()
);

-- Vista útil: predicciones de hoy con detalle de batting order
CREATE OR REPLACE VIEW v_today_predictions AS
SELECT
    p.game_date,
    p.game_pk,
    p.team_abbr,
    p.side,
    p.expected_runs,
    p.model_version,
    batter->>'name'       AS batter_name,
    (batter->>'slot')::INT AS slot,
    (batter->>'ev_per_pa')::FLOAT AS ev_per_pa,
    (batter->>'prob_hr')::FLOAT   AS prob_hr,
    (batter->>'prob_k')::FLOAT    AS prob_k,
    (batter->>'prob_bb')::FLOAT   AS prob_bb
FROM predictions p,
     jsonb_array_elements(p.batting_order) AS batter
WHERE p.game_date = CURRENT_DATE;

COMMENT ON TABLE predictions IS 'Predicciones diarias del modelo MLB AI. Upsert idempotente por (game_date, game_pk, team_abbr, side).';
COMMENT ON TABLE shadow_predictions IS 'Predicciones del modelo challenger en shadow mode. Para comparación offline vs champion.';
COMMENT ON TABLE drift_reports IS 'Historial de PSI calculado por el DAG diario. Usado para tracking de degradación del modelo.';
