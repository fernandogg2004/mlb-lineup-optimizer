#!/usr/bin/env bash
# =============================================================================
# deploy/first_run.sh
# Primer arranque del stack MLB AI en el VPS.
# Ejecutar después de setup_vps.sh y de editar .env.
#
# Uso (desde /opt/mlb-ai):
#   bash deploy/first_run.sh
#
# Qué hace:
#   1. Construye imágenes Docker y arranca todos los contenedores
#   2. Espera a que Airflow esté listo
#   3. Crea la conexión mlb_predictions_db
#   4. Crea las variables de Airflow
#   5. Crea las tablas en PostgreSQL
#   6. Despausa el DAG mlb_gameday_orchestrator
# =============================================================================
set -euo pipefail

[[ -f .env ]] || { echo "ERROR: Falta el archivo .env. Copia .env.vps.example y edítalo."; exit 1; }
source .env

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] $*${NC}"; }
step() { echo -e "\n${YELLOW}=== $* ===${NC}"; }

COMPOSE="docker compose -f docker-compose.vps.yml --env-file .env"

# ---------------------------------------------------------------------------
step "[0/6] Preparando directorios con permisos correctos"
mkdir -p logs reports/gameday
chown -R 50000:0 logs reports
log "Directorios listos."

# ---------------------------------------------------------------------------
step "[1/6] Construyendo imágenes y arrancando contenedores"
$COMPOSE up -d --build
log "Contenedores arrancados. Esperando que estén healthy..."

# ---------------------------------------------------------------------------
step "[2/6] Esperando que Airflow webserver esté listo"
MAX_WAIT=180
ELAPSED=0
until $COMPOSE exec -T airflow-webserver curl -sf http://localhost:8080/health >/dev/null 2>&1; do
  if (( ELAPSED >= MAX_WAIT )); then
    echo "ERROR: Airflow no respondió en ${MAX_WAIT}s."
    $COMPOSE logs --tail=30 airflow-webserver
    exit 1
  fi
  echo "  → Esperando webserver... (${ELAPSED}s)"
  sleep 10
  (( ELAPSED += 10 ))
done
log "Airflow webserver listo."

# ---------------------------------------------------------------------------
step "[3/6] Creando conexión Airflow: mlb_predictions_db"
$COMPOSE exec -T airflow-webserver \
  airflow connections add mlb_predictions_db \
    --conn-type postgres \
    --conn-host postgres-mlb \
    --conn-port 5432 \
    --conn-schema "${POSTGRES_DB:-mlb_predictions}" \
    --conn-login  "${POSTGRES_USER:-mlb}" \
    --conn-password "${POSTGRES_PASSWORD}" \
  2>&1 | grep -v FutureWarning | grep -v "configuration.py" || log "(conexión ya existía)"

# ---------------------------------------------------------------------------
step "[4/6] Creando variables de Airflow"
_var() {
  $COMPOSE exec -T airflow-webserver airflow variables set "$1" "$2" \
    2>&1 | grep -v FutureWarning | grep -v "configuration.py" | tail -1
}
_var MLB_API_INFERENCE_URL     "http://mlb_api:8000"
_var GAMEDAY_REPORTS_DIR       "/opt/airflow/reports/gameday"
_var SLACK_WEBHOOK_URL         "${SLACK_WEBHOOK_URL:-}"
log "Variables creadas."

# ---------------------------------------------------------------------------
step "[5/6] Creando tablas en PostgreSQL"
$COMPOSE exec -T postgres-mlb \
  psql -U "${POSTGRES_USER:-mlb}" -d "${POSTGRES_DB:-mlb_predictions}" << 'SQL'
CREATE TABLE IF NOT EXISTS gameday_predictions (
    id                       SERIAL PRIMARY KEY,
    game_pk                  INTEGER NOT NULL,
    game_date                DATE NOT NULL,
    home_team_id             INTEGER,
    away_team_id             INTEGER,
    home_team_name           VARCHAR(100),
    away_team_name           VARCHAR(100),
    home_batting_order       INTEGER[],
    away_batting_order       INTEGER[],
    home_starting_pitcher_id INTEGER,
    away_starting_pitcher_id INTEGER,
    ai_recommended_lineup    INTEGER[],
    ai_expected_runs         FLOAT,
    win_probability          FLOAT,
    optimization_mode        VARCHAR(10),
    model_version            VARCHAR(50),
    lineup_source            VARCHAR(20),
    predicted_at             TIMESTAMPTZ DEFAULT NOW(),
    actual_home_runs         INTEGER,
    actual_away_runs         INTEGER,
    delta_er                 FLOAT,
    game_state               VARCHAR(30),
    resolved_at              TIMESTAMPTZ,
    UNIQUE (game_pk)
);

CREATE TABLE IF NOT EXISTS predictions (
    id            SERIAL PRIMARY KEY,
    game_date     DATE NOT NULL,
    game_pk       INTEGER NOT NULL,
    team_abbr     VARCHAR(10) NOT NULL,
    side          VARCHAR(5)  NOT NULL,
    batting_order JSONB,
    expected_runs FLOAT,
    model_version VARCHAR(50),
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ,
    UNIQUE (game_date, game_pk, team_abbr, side)
);
SQL
log "Tablas creadas."

# ---------------------------------------------------------------------------
step "[6/6] Activando DAG mlb_gameday_orchestrator"
$COMPOSE exec -T airflow-webserver \
  airflow dags unpause mlb_gameday_orchestrator \
  2>&1 | grep -v FutureWarning | grep -v "configuration.py" | tail -2
log "DAG activo. Se ejecutará automáticamente cada día a las 08:00 UTC."

# ---------------------------------------------------------------------------
PUBLIC_IP=$(curl -s --max-time 5 ifconfig.me 2>/dev/null || echo "<IP_DEL_SERVIDOR>")

echo ""
echo -e "${GREEN}=================================================================${NC}"
echo -e "${GREEN}  Despliegue completado.${NC}"
echo ""
echo -e "  API:     http://${PUBLIC_IP}:8000/docs"
echo -e "  Airflow: http://${PUBLIC_IP}:8080  (admin / \$AIRFLOW_ADMIN_PASSWORD)"
echo ""
echo -e "  Reportes post-partido:"
echo -e "    /opt/mlb-ai/reports/gameday/gameday_report_YYYY-MM-DD.pdf"
echo ""
echo -e "  Ver logs en tiempo real:"
echo -e "    docker compose -f docker-compose.vps.yml logs -f"
echo ""
echo -e "  Actualizar a nueva versión:"
echo -e "    bash deploy/update.sh"
echo -e "${GREEN}=================================================================${NC}"
