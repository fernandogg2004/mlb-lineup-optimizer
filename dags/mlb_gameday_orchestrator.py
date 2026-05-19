"""
dags/mlb_gameday_orchestrator.py
=================================================================================================
Game-Day Orchestrator — Motor de operaciones en tiempo real del MLB Lineup Optimizer.

Programación: 08:00 UTC diario (10:00 Madrid / 04:00 EDT).

Flujo de 3 tareas con Dynamic Task Mapping:

    Task 1 — fetch_todays_schedule
        Consulta la MLB StatsAPI y devuelve la lista de partidos del día.
        Output: list[dict] — una entrada por gamePk.

    Task 2 — game_pipeline_dynamic_map (×N instancias, una por partido)
        Por cada partido en paralelo:
            2a. Espera la ventana T-2h antes del primer pitch.
            2b. LineupSensor: consulta el live-feed de la MLB hasta que el lineup
                oficial esté disponible.  Si no llega → falla con retries=4 / 30min.
                Si el partido ya empezó sin lineup oficial → usa roster proyectado
                (fallback) para no bloquear el pipeline.
            2c. Pre-game Inference: construye el request de optimización con las
                stats de los bateadores confirmados, llama a /v1/optimize/lineup y
                guarda la predicción en PostgreSQL (tabla gameday_predictions).

    Task 3 — await_and_resolve_all_games (trigger_rule=ALL_DONE)
        Corre cuando TODAS las Task 2 han terminado (éxito o fallo).
        Llama a PostGameEvaluator.run():
            - Espera polling hasta que todos los partidos alcancen estado Final.
            - Descarga boxscores reales.
            - Calcula delta E[R], MAE, RMSE.
            - Genera informe PDF con reportlab.
            - Guarda resoluciones en PostgreSQL.
            - Envía resumen a Slack.

Zonas horarias:
    - Toda la lógica interna usa UTC (pytz.UTC).
    - La MLB StatsAPI devuelve gameDate en ISO 8601 UTC.
    - Conversiones a ET solo para logging y presentación.

Manejo de errores:
    - Cada llamada HTTP tiene reintentos exponenciales (_http_get_json).
    - El LineupSensor falla intencionalmente (raises AirflowException) para
      activar los reintentos de Airflow (retries=4, retry_delay=30min).
    - Si el partido ya inició (abstractGameState == "Live") y el lineup sigue
      vacío, se usa el roster 40-Man proyectado con stats de carrera como fallback.
    - Task 3 usa trigger_rule=ALL_DONE para ejecutarse incluso si alguna Task 2 falló.

Notas de producción:
    - El bloque de espera en Task 2 usa time.sleep() en chunks de 60 segundos.
      En producción con muchos partidos simultáneos, reemplazar el bloque de espera
      por un DateTimeSensor(mode="reschedule") para liberar el worker slot durante
      la espera larga. Ejemplo al final del archivo.
    - La llamada a la FastAPI usa la URL de Airflow Variable MLB_API_INFERENCE_URL.
      En entornos con ECS/K8s, usar el endpoint interno del servicio.
    - Las credenciales PostgreSQL se leen de la Airflow Connection "mlb_predictions_db".
    - La tabla gameday_predictions debe existir antes de ejecutar el DAG.
      DDL disponible en el docstring de _ensure_predictions_table().
=================================================================================================
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import pendulum
import psycopg2
import psycopg2.extras
import pytz
import requests
import structlog
from airflow.decorators import dag, task
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.models import Variable
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)
slog = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuración central — variables de entorno / Airflow Variables
# ---------------------------------------------------------------------------

_MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
_MLB_API_V1_1 = "https://statsapi.mlb.com/api/v1.1"
_SPORT_ID = 1

_UTC = pytz.UTC
_ET_TZ = pytz.timezone("America/New_York")

# Ventana pre-partido: 2 horas antes del primer pitch para detectar el lineup
_LINEUP_WINDOW_HOURS = 2

# Si el lineup no llega en este tiempo desde T-2h → el partido ya empezó; usar fallback
_LINEUP_FALLBACK_AFTER_GAME_START = True

# Intervalo entre polls del LineupSensor (cuando está en la ventana de 2 horas)
_LINEUP_POLL_SECONDS = 300  # 5 minutos

# Sleep granular dentro del bloque de espera pre-ventana (chunk para heartbeat de Airflow)
_WAIT_CHUNK_SECONDS = 60

# Reintentos HTTP para llamadas a la MLB API y a la FastAPI interna
_HTTP_RETRIES = 3
_HTTP_BACKOFF = 2.0

# Timeout de cada request HTTP
_HTTP_TIMEOUT = 30

# Nombre de la conexión PostgreSQL en Airflow
_PG_CONN_ID = "mlb_predictions_db"

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "mlb-gameday",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": Variable.get("ALERT_EMAIL", default_var="mlops@yourorg.com"),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ---------------------------------------------------------------------------
# Helpers HTTP (fuera del DAG — reutilizables, sin estado de Airflow)
# ---------------------------------------------------------------------------


def _http_get_json(url: str, params: dict | None = None) -> dict:
    """GET con reintentos exponenciales ante error de red o 5xx."""
    for attempt in range(1, _HTTP_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if code < 500:
                raise  # 4xx no reintentable
            if attempt == _HTTP_RETRIES:
                raise
            wait = _HTTP_BACKOFF ** attempt
            log.warning("MLB API HTTP %d retry %d/%d → %.0fs", code, attempt, _HTTP_RETRIES, wait)
            time.sleep(wait)
        except requests.exceptions.RequestException as exc:
            if attempt == _HTTP_RETRIES:
                raise
            wait = _HTTP_BACKOFF ** attempt
            log.warning("Request error retry %d/%d → %.0fs: %s", attempt, _HTTP_RETRIES, wait, exc)
            time.sleep(wait)
    raise RuntimeError(f"_http_get_json exhausted retries for {url}")  # inalcanzable


def _http_post_json(url: str, payload: dict, headers: dict | None = None) -> dict:
    """POST JSON con reintentos exponenciales ante 5xx."""
    headers = headers or {"Content-Type": "application/json"}
    for attempt in range(1, _HTTP_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if code < 500:
                raise
            if attempt == _HTTP_RETRIES:
                raise
            time.sleep(_HTTP_BACKOFF ** attempt)
        except requests.exceptions.RequestException as exc:
            if attempt == _HTTP_RETRIES:
                raise
            time.sleep(_HTTP_BACKOFF ** attempt)
    raise RuntimeError(f"_http_post_json exhausted retries for {url}")


# ---------------------------------------------------------------------------
# Helpers de lógica de negocio (sin estado de Airflow)
# ---------------------------------------------------------------------------


def _parse_game_dt_utc(game_date_str: str) -> datetime:
    """Convierte el gameDate de la MLB API (ISO 8601 UTC) en un datetime timezone-aware."""
    # Formato: "2026-05-19T18:10:00Z" o "2026-05-19T18:10:00.000Z"
    clean = game_date_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(clean)
    if dt.tzinfo is None:
        dt = _UTC.localize(dt)
    return dt.astimezone(_UTC)


def _fetch_schedule_for_date(target_date: str) -> list[dict]:
    """Obtiene la lista de partidos de la MLB para una fecha dada.

    Args:
        target_date: Fecha en formato YYYY-MM-DD.

    Returns:
        Lista de dicts con game_pk, game_date_utc, home/away team info y status.
    """
    data = _http_get_json(
        f"{_MLB_API_BASE}/schedule",
        params={
            "sportId": _SPORT_ID,
            "date": target_date,
            "hydrate": "team,venue,status",
        },
    )
    games: list[dict] = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            game_pk = game.get("gamePk")
            game_date_str = game.get("gameDate")
            if not game_pk or not game_date_str:
                continue

            status = game.get("status", {})
            abstract_state = status.get("abstractGameState", "Preview")
            detailed_state = status.get("detailedState", "Scheduled")

            home_team = game.get("teams", {}).get("home", {}).get("team", {})
            away_team = game.get("teams", {}).get("away", {}).get("team", {})

            games.append({
                "game_pk": int(game_pk),
                "game_date_utc": game_date_str,
                "home_team_id": home_team.get("id"),
                "away_team_id": away_team.get("id"),
                "home_team_name": home_team.get("name", "Unknown"),
                "away_team_name": away_team.get("name", "Unknown"),
                "home_team_abbr": home_team.get("abbreviation", "UNK"),
                "away_team_abbr": away_team.get("abbreviation", "UNK"),
                "venue_name": game.get("venue", {}).get("name", ""),
                "abstract_state": abstract_state,
                "detailed_state": detailed_state,
            })

    log.info("Schedule fetched for %s: %d games found.", target_date, len(games))
    return games


def _sense_lineup(game_pk: int) -> tuple[bool, str, dict]:
    """Consulta el live-feed de un partido y detecta si el lineup oficial está disponible.

    Returns:
        (lineup_available: bool, abstract_game_state: str, lineup_data: dict)
        lineup_data contiene:
            home_batting_order, away_batting_order,
            home_starting_pitcher_id, away_starting_pitcher_id
    """
    data = _http_get_json(f"{_MLB_API_V1_1}/game/{game_pk}/feed/live")

    game_state = (
        data.get("gameData", {})
        .get("status", {})
        .get("abstractGameState", "Preview")
    )

    teams = data.get("liveData", {}).get("boxscore", {}).get("teams", {})
    home = teams.get("home", {})
    away = teams.get("away", {})

    home_order: list[int] = home.get("battingOrder", [])
    away_order: list[int] = away.get("battingOrder", [])
    home_pitchers: list[int] = home.get("pitchers", [])
    away_pitchers: list[int] = away.get("pitchers", [])

    # También chequear probablePitchers en gameData si el boxscore está vacío
    if not home_pitchers or not away_pitchers:
        probable = data.get("gameData", {}).get("probablePitchers", {})
        if not home_pitchers and probable.get("home", {}).get("id"):
            home_pitchers = [probable["home"]["id"]]
        if not away_pitchers and probable.get("away", {}).get("id"):
            away_pitchers = [probable["away"]["id"]]

    lineup_ready = (
        len(home_order) >= 9
        and len(away_order) >= 9
        and bool(home_pitchers)
        and bool(away_pitchers)
    )

    lineup_data = {
        "home_batting_order": home_order[:9],
        "away_batting_order": away_order[:9],
        "home_starting_pitcher_id": home_pitchers[0] if home_pitchers else None,
        "away_starting_pitcher_id": away_pitchers[0] if away_pitchers else None,
    }

    return lineup_ready, game_state, lineup_data


def _fetch_player_career_stats(player_id: int) -> dict | None:
    """Obtiene estadísticas de carrera de un bateador desde la MLB Stats API.

    Returns:
        Dict con avg, obp, slg, iso, woba_approx, bat_side, full_name.
        None si el jugador no tiene estadísticas.
    """
    try:
        data = _http_get_json(
            f"{_MLB_API_BASE}/people/{player_id}",
            params={"hydrate": "stats(group=[hitting],type=[career])"},
        )
        people = data.get("people", [])
        if not people:
            return None

        person = people[0]
        bat_side = person.get("batSide", {}).get("code", "R")
        full_name = person.get("fullName", f"Player {player_id}")

        stats_list = person.get("stats", [])
        career_stats: dict = {}
        for s in stats_list:
            if s.get("type", {}).get("displayName") == "career":
                splits = s.get("splits", [])
                if splits:
                    career_stats = splits[0].get("stat", {})
                    break

        if not career_stats:
            # Usar promedios de liga si no hay stats de carrera
            return {
                "player_id": player_id,
                "full_name": full_name,
                "bat_side": bat_side,
                "avg": 0.248,
                "obp": 0.318,
                "slg": 0.398,
                "iso": 0.150,
                "woba_approx": 0.315,
                "source": "league_average_fallback",
            }

        avg = float(career_stats.get("avg", "0.248").lstrip(".") and career_stats.get("avg", 0.248) or 0.248)
        obp = float(career_stats.get("obp", 0.318) or 0.318)
        slg = float(career_stats.get("slg", 0.398) or 0.398)
        iso = round(max(slg - avg, 0.0), 3)
        # wOBA aproximado: 0.72*BB + 0.75*HBP + 0.9*1B + 1.25*2B + 1.6*3B + 2.1*HR
        # Simplificado: (OBP * 1.2 + SLG * 0.7) / 2
        woba_approx = round((obp * 1.2 + slg * 0.7) / 2, 3)

        return {
            "player_id": player_id,
            "full_name": full_name,
            "bat_side": bat_side,
            "avg": float(avg),
            "obp": float(obp),
            "slg": float(slg),
            "iso": iso,
            "woba_approx": woba_approx,
            "source": "mlb_career_stats",
        }
    except Exception as exc:
        log.warning("Career stats fetch failed for player_id=%s: %s", player_id, exc)
        return None


def _approximate_prob_vector(stats: dict) -> list[float]:
    """Genera un prob_vector de 7 clases a partir de estadísticas de carrera.

    Clases: [OUT_IN_PLAY, STRIKEOUT, WALK_HBP, SINGLE, DOUBLE, TRIPLE, HOME_RUN]

    NOTA DE PRODUCCIÓN:
        Esta función usa stats de carrera como proxy. En producción, el prob_vector
        debe calcularse con el modelo AtBatPredictor entrenado sobre features Statcast
        (rolling windows, platoon splits, park factors). Aquí se usa como fallback
        cuando el feature store no está disponible en el worker de Airflow.
    """
    avg = float(stats.get("avg", 0.248))
    obp = float(stats.get("obp", 0.318))
    slg = float(stats.get("slg", 0.398))

    # Tasas aproximadas por PA
    pa_adj = 0.97  # PA ≈ AB * 1.03, ajuste simple
    p_bb_hbp = max(round((obp - avg) * pa_adj, 4), 0.04)
    p_hr = max(round((slg - avg) / 4.2, 4), 0.01)  # muy rough
    p_triple = 0.005
    p_double = max(round((slg - avg - p_hr * 3 - p_triple * 2) * 0.35, 4), 0.02)
    p_single = max(round(avg - p_hr - p_triple - p_double, 4), 0.08)
    # K rate correlaciona negativamente con OBP
    p_k = max(round(0.22 - (obp - 0.310) * 0.6, 4), 0.08)
    p_out = max(round(1.0 - p_bb_hbp - p_hr - p_triple - p_double - p_single - p_k, 4), 0.20)

    raw = [p_out, p_k, p_bb_hbp, p_single, p_double, p_triple, p_hr]
    total = sum(raw)
    return [round(x / total, 4) for x in raw]


def _build_optimize_request(game_info: dict, lineup_data: dict) -> dict | None:
    """Construye el payload para POST /v1/optimize/lineup con el lineup confirmado.

    Obtiene stats de carrera para cada bateador del home team y construye un
    PlayerRosterEntry por cada uno. Usa el lanzador abridor del away team como rival.

    Returns:
        Payload dict compatible con LineupOptimizeRequest, o None si no se pudo
        construir el roster completo (< 9 jugadores con stats).
    """
    home_order = lineup_data.get("home_batting_order", [])
    away_pitcher_id = lineup_data.get("away_starting_pitcher_id")

    if len(home_order) < 9:
        log.warning(
            "build_optimize_request: game_pk=%s — home batting order has only %d players",
            game_info["game_pk"], len(home_order)
        )
        return None

    # Construir roster de bateadores
    roster: list[dict] = []
    for pid in home_order[:9]:
        stats = _fetch_player_career_stats(int(pid))
        if stats is None:
            log.warning("No stats for player_id=%s, using league average", pid)
            stats = {
                "player_id": pid,
                "full_name": f"Player {pid}",
                "bat_side": "R",
                "avg": 0.248, "obp": 0.318, "slg": 0.398,
                "iso": 0.150, "woba_approx": 0.315,
                "source": "league_average_fallback",
            }
        prob_vector = _approximate_prob_vector(stats)
        roster.append({
            "player_id": int(pid),
            "player_name": stats["full_name"],
            "obp": round(stats["obp"], 3),
            "woba": round(stats["woba_approx"], 3),
            "iso": round(stats["iso"], 3),
            "batter_stand": stats["bat_side"],
            "prob_vector": prob_vector,
        })

    if len(roster) < 9:
        log.warning("Could not build 9-player roster for game_pk=%s", game_info["game_pk"])
        return None

    # Información del lanzador rival (away pitcher)
    rival_pitcher: dict = {
        "pitcher_id": away_pitcher_id or 0,
        "pitcher_name": f"Pitcher {away_pitcher_id}",
        "hand": "R",  # default; en producción: fetch de la MLB API
        "era": 4.00,
        "pitch_mix": {"FF": 0.55, "SL": 0.20, "CH": 0.15, "CB": 0.10},
    }

    if away_pitcher_id:
        try:
            pitcher_data = _http_get_json(f"{_MLB_API_BASE}/people/{away_pitcher_id}")
            people = pitcher_data.get("people", [])
            if people:
                p = people[0]
                rival_pitcher["pitcher_name"] = p.get("fullName", rival_pitcher["pitcher_name"])
                rival_pitcher["hand"] = p.get("pitchHand", {}).get("code", "R")
        except Exception as exc:
            log.warning("Failed to fetch pitcher info for id=%s: %s", away_pitcher_id, exc)

    game_id = (
        f"{game_info['game_date_utc'][:10]}-"
        f"{game_info['away_team_abbr']}-"
        f"{game_info['home_team_abbr']}"
    )

    return {
        "roster": roster,
        "rival_pitcher": rival_pitcher,
        "game_id": game_id,
        "fast_mode": True,  # Pre-game: fast mode (3s) para todos los partidos en paralelo
        "apply_feedback_overrides": False,
    }


def _save_prediction_to_db(
    pg_hook: PostgresHook,
    game_info: dict,
    lineup_data: dict,
    prediction: dict,
    lineup_source: str,
) -> None:
    """Persiste la predicción pre-partido en la tabla gameday_predictions.

    DDL requerido (ejecutar una vez antes del primer run del DAG):

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
    """
    upsert_sql = """
        INSERT INTO gameday_predictions (
            game_pk, game_date,
            home_team_id, away_team_id,
            home_team_name, away_team_name,
            home_batting_order, away_batting_order,
            home_starting_pitcher_id, away_starting_pitcher_id,
            ai_recommended_lineup, ai_expected_runs,
            win_probability, optimization_mode,
            model_version, lineup_source, predicted_at
        ) VALUES (
            %(game_pk)s, %(game_date)s,
            %(home_team_id)s, %(away_team_id)s,
            %(home_team_name)s, %(away_team_name)s,
            %(home_batting_order)s, %(away_batting_order)s,
            %(home_starting_pitcher_id)s, %(away_starting_pitcher_id)s,
            %(ai_recommended_lineup)s, %(ai_expected_runs)s,
            %(win_probability)s, %(optimization_mode)s,
            %(model_version)s, %(lineup_source)s, NOW()
        )
        ON CONFLICT (game_pk) DO UPDATE SET
            ai_recommended_lineup    = EXCLUDED.ai_recommended_lineup,
            ai_expected_runs         = EXCLUDED.ai_expected_runs,
            win_probability          = EXCLUDED.win_probability,
            lineup_source            = EXCLUDED.lineup_source,
            predicted_at             = NOW()
    """
    params = {
        "game_pk": game_info["game_pk"],
        "game_date": game_info["game_date_utc"][:10],
        "home_team_id": game_info["home_team_id"],
        "away_team_id": game_info["away_team_id"],
        "home_team_name": game_info["home_team_name"],
        "away_team_name": game_info["away_team_name"],
        "home_batting_order": lineup_data.get("home_batting_order", []),
        "away_batting_order": lineup_data.get("away_batting_order", []),
        "home_starting_pitcher_id": lineup_data.get("home_starting_pitcher_id"),
        "away_starting_pitcher_id": lineup_data.get("away_starting_pitcher_id"),
        "ai_recommended_lineup": prediction.get("lineup_player_ids", []),
        "ai_expected_runs": prediction.get("expected_runs", 0.0),
        "win_probability": prediction.get("win_probability", 0.0),
        "optimization_mode": prediction.get("optimization_mode", "fast"),
        "model_version": prediction.get("model_version", "unknown"),
        "lineup_source": lineup_source,
    }
    conn = pg_hook.get_conn()
    cur = conn.cursor()
    cur.execute(upsert_sql, params)
    conn.commit()
    slog.info(
        "prediction_saved_to_db",
        game_pk=game_info["game_pk"],
        ai_expected_runs=params["ai_expected_runs"],
        lineup_source=lineup_source,
    )


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------


@dag(
    dag_id="mlb_gameday_orchestrator",
    description=(
        "Game-Day Orchestrator: lineup sensing T-2h → pre-game inference → "
        "post-game delta E[R] evaluation. Dynamic Task Mapping (1 instancia/partido)."
    ),
    schedule="0 8 * * *",  # 08:00 UTC = 10:00 Madrid = 04:00 EDT
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlb", "gameday", "realtime", "dynamic-mapping"],
    doc_md=__doc__,
)
def mlb_gameday_orchestrator():
    """DAG del Game-Day Orchestrator del MLB Lineup Optimizer."""

    # ===================================================================
    # TASK 1 — Fetch Today's Schedule
    # ===================================================================

    @task(task_id="fetch_todays_schedule")
    def fetch_todays_schedule(**context) -> list[dict]:
        """Obtiene los partidos MLB programados para hoy desde la StatsAPI.

        Usa la logical_date del DAG run como fecha objetivo para garantizar
        idempotencia en backfills.

        Returns:
            Lista de dicts con game_pk, game_date_utc, team info y estado.
            Lista vacía si no hay partidos hoy (día sin MLB).
        """
        logical_date: pendulum.DateTime = context["logical_date"]
        target_date = logical_date.in_timezone("UTC").strftime("%Y-%m-%d")

        slog.info(
            "fetch_schedule_start",
            target_date=target_date,
            dag_run_id=context["run_id"],
        )

        try:
            games = _fetch_schedule_for_date(target_date)
        except Exception as exc:
            slog.error("fetch_schedule_failed", target_date=target_date, error=str(exc))
            raise AirflowException(f"MLB StatsAPI schedule fetch failed: {exc}") from exc

        if not games:
            slog.info("no_games_today", target_date=target_date)
            # Lanzar AirflowSkipException vacía la lista y las tareas downstream
            # se marcan como Skipped en el UI de Airflow
            raise AirflowSkipException(f"No MLB games scheduled for {target_date}")

        slog.info(
            "schedule_fetched",
            target_date=target_date,
            n_games=len(games),
            game_pks=[g["game_pk"] for g in games],
            first_pitch_utc=min(g["game_date_utc"] for g in games),
        )
        return games

    # ===================================================================
    # TASK 2 — Game Pipeline (Dynamic Map — una instancia por partido)
    # ===================================================================

    @task(
        task_id="game_pipeline_dynamic_map",
        retries=4,
        retry_delay=timedelta(minutes=30),
        # Importante: execution_timeout generoso para partidos de tarde/noche.
        # Un partido de Costa Oeste puede empezar a las 22:10 UTC; la tarea
        # podría esperar hasta las 20:10 UTC → ~12 horas de espera máxima.
        # En producción: reemplazar el bloque de espera por DateTimeSensor(mode="reschedule").
        execution_timeout=timedelta(hours=15),
    )
    def game_pipeline_dynamic_map(game_info: dict, **context) -> dict:
        """Flujo completo pre-partido para un único gamePk.

        Etapas internas:
            1. Calcular ventana T-2h y esperar hasta ella.
            2. LineupSensor: consultar el lineup oficial.
               → Si no está: raise AirflowException → retry en 30 min.
               → Si el partido ya empezó sin lineup: usar roster proyectado.
            3. Construir request de optimización con lineup confirmado.
            4. Llamar a /v1/optimize/lineup en la FastAPI interna.
            5. Guardar predicción en PostgreSQL.

        Args:
            game_info: Dict producido por fetch_todays_schedule.

        Returns:
            Dict con game_pk, lineup_source, ai_expected_runs, status.
        """
        game_pk: int = game_info["game_pk"]
        game_dt_utc: datetime = _parse_game_dt_utc(game_info["game_date_utc"])
        game_dt_et = game_dt_utc.astimezone(_ET_TZ)
        matchup = f"{game_info['away_team_name']} @ {game_info['home_team_name']}"
        now_utc = datetime.now(_UTC)

        slog.info(
            "game_pipeline_start",
            game_pk=game_pk,
            matchup=matchup,
            first_pitch_utc=game_dt_utc.strftime("%Y-%m-%d %H:%M UTC"),
            first_pitch_et=game_dt_et.strftime("%H:%M ET"),
            map_index=context.get("map_index_template", "?"),
        )

        # ------------------------------------------------------------------
        # Etapa 2a: Espera hasta la ventana T-2h
        # ------------------------------------------------------------------
        window_start_utc = game_dt_utc - timedelta(hours=_LINEUP_WINDOW_HOURS)
        now_utc = datetime.now(_UTC)

        if now_utc < window_start_utc:
            wait_seconds = (window_start_utc - now_utc).total_seconds()
            slog.info(
                "waiting_for_pregame_window",
                game_pk=game_pk,
                window_start_utc=window_start_utc.strftime("%H:%M UTC"),
                wait_minutes=round(wait_seconds / 60, 1),
            )
            # NOTA DE PRODUCCIÓN:
            # Reemplazar este bucle por DateTimeSensor para liberar el worker:
            #
            #   from airflow.sensors.time_sensor import TimeSensor
            #   TimeSensor(
            #       task_id=f"wait_window_{game_pk}",
            #       target_time=window_start_utc.time(),
            #       mode="reschedule",
            #   )
            #
            # Con time.sleep() en chunks de 60s el worker queda ocupado
            # pero funciona correctamente en entornos de worker dedicado.
            while datetime.now(_UTC) < window_start_utc:
                remaining = (window_start_utc - datetime.now(_UTC)).total_seconds()
                time.sleep(min(_WAIT_CHUNK_SECONDS, max(remaining, 1)))

        slog.info("pregame_window_reached", game_pk=game_pk, matchup=matchup)

        # ------------------------------------------------------------------
        # Etapa 2b: LineupSensor
        # ------------------------------------------------------------------
        lineup_available, game_state, lineup_data = _sense_lineup(game_pk)

        lineup_source: str
        if lineup_available:
            lineup_source = "official"
            slog.info(
                "lineup_confirmed",
                game_pk=game_pk,
                matchup=matchup,
                source=lineup_source,
                home_order=lineup_data["home_batting_order"],
                home_starter=lineup_data["home_starting_pitcher_id"],
            )
        elif game_state in ("Live", "In Progress", "Final", "Game Over"):
            # El partido ya comenzó o terminó sin lineup oficial publicado
            if not _LINEUP_FALLBACK_AFTER_GAME_START:
                slog.warning(
                    "lineup_never_published_skipping",
                    game_pk=game_pk,
                    game_state=game_state,
                )
                return {
                    "game_pk": game_pk,
                    "matchup": matchup,
                    "lineup_source": "skip",
                    "ai_expected_runs": None,
                    "status": "skipped_no_lineup",
                }

            # Fallback: usar el roster 40-Man del home team
            slog.warning(
                "lineup_fallback_to_projected",
                game_pk=game_pk,
                game_state=game_state,
                matchup=matchup,
            )
            lineup_data = _build_projected_lineup(
                game_info["home_team_id"],
                game_info["away_team_id"],
            )
            lineup_source = "projected"

        else:
            # Lineup aún no publicado y el partido no ha comenzado → reintento
            slog.warning(
                "lineup_not_available_triggering_retry",
                game_pk=game_pk,
                game_state=game_state,
                matchup=matchup,
                retries_remaining=context["task_instance"].max_tries - context["task_instance"].try_number,
            )
            raise AirflowException(
                f"game_pk={game_pk} ({matchup}): lineup not published yet "
                f"(state={game_state}). Retry in 30 min."
            )

        # ------------------------------------------------------------------
        # Etapa 2c: Pre-game Inference
        # ------------------------------------------------------------------
        optimize_payload = _build_optimize_request(game_info, lineup_data)
        if optimize_payload is None:
            slog.warning(
                "optimize_request_build_failed",
                game_pk=game_pk,
                matchup=matchup,
                reason="Insufficient player stats",
            )
            return {
                "game_pk": game_pk,
                "matchup": matchup,
                "lineup_source": lineup_source,
                "ai_expected_runs": None,
                "status": "inference_skipped",
            }

        api_url = Variable.get("MLB_API_INFERENCE_URL", default_var="http://api:8000")

        try:
            prediction = _http_post_json(
                f"{api_url}/v1/optimize/lineup",
                payload=optimize_payload,
            )
        except Exception as exc:
            slog.error(
                "preinference_api_call_failed",
                game_pk=game_pk,
                matchup=matchup,
                error=str(exc),
            )
            # Fallo de inferencia no es crítico para el pipeline general — log + continúa
            return {
                "game_pk": game_pk,
                "matchup": matchup,
                "lineup_source": lineup_source,
                "ai_expected_runs": None,
                "status": f"inference_failed: {exc}",
            }

        slog.info(
            "preinference_complete",
            game_pk=game_pk,
            matchup=matchup,
            ai_expected_runs=prediction.get("expected_runs"),
            win_probability=prediction.get("win_probability"),
            model_version=prediction.get("model_version"),
            lineup_source=lineup_source,
        )

        # ------------------------------------------------------------------
        # Etapa 2d: Guardar en PostgreSQL
        # ------------------------------------------------------------------
        pg_hook = PostgresHook(postgres_conn_id=_PG_CONN_ID)
        try:
            _save_prediction_to_db(pg_hook, game_info, lineup_data, prediction, lineup_source)
        except Exception as exc:
            slog.error(
                "save_prediction_failed",
                game_pk=game_pk,
                error=str(exc),
            )
            # No bloqueamos el pipeline si falla la BD; el error se propaga en Task 3
            raise AirflowException(f"game_pk={game_pk}: DB save failed: {exc}") from exc

        return {
            "game_pk": game_pk,
            "matchup": matchup,
            "lineup_source": lineup_source,
            "ai_expected_runs": prediction.get("expected_runs"),
            "win_probability": prediction.get("win_probability"),
            "model_version": prediction.get("model_version"),
            "status": "ok",
        }

    # ===================================================================
    # TASK 3 — Post-game Evaluation (Join dinámico)
    # ===================================================================

    @task(
        task_id="await_and_resolve_all_games",
        trigger_rule=TriggerRule.ALL_DONE,  # corre aunque alguna Task 2 haya fallado
        retries=2,
        retry_delay=timedelta(minutes=15),
        execution_timeout=timedelta(hours=10),
    )
    def await_and_resolve_all_games(games: list[dict], **context) -> dict:
        """Evaluación analítica post-partido de todos los juegos del día.

        Se ejecuta cuando TODAS las instancias de game_pipeline_dynamic_map
        han terminado (éxito o fallo). Usa PostGameEvaluator para:
            - Esperar que todos los partidos lleguen a estado Final.
            - Descargar boxscores y cruzar con predicciones en DB.
            - Calcular delta E[R], MAE, RMSE.
            - Generar PDF con reportlab.
            - Guardar resoluciones en PostgreSQL.
            - Notificar por Slack.

        Args:
            games: Lista de dicts con info de los partidos del día
                   (output de fetch_todays_schedule).

        Returns:
            Dict con DayMetrics serializado.
        """
        from src.orchestration.post_game_evaluator import PostGameEvaluator

        game_pks = [g["game_pk"] for g in games]
        game_date = games[0]["game_date_utc"][:10] if games else context["ds"]

        slog.info(
            "post_game_eval_start",
            game_date=game_date,
            game_pks=game_pks,
            dag_run_id=context["run_id"],
        )

        postgres_dsn = _build_postgres_dsn()
        slack_webhook = Variable.get("SLACK_WEBHOOK_URL", default_var="")
        reports_dir = Variable.get("GAMEDAY_REPORTS_DIR", default_var="reports/gameday")

        evaluator = PostGameEvaluator(
            postgres_dsn=postgres_dsn,
            slack_webhook_url=slack_webhook,
            reports_dir=reports_dir,
        )

        try:
            metrics = evaluator.run(game_pks=game_pks, game_date=game_date)
        except Exception as exc:
            slog.error(
                "post_game_eval_failed",
                game_date=game_date,
                error=str(exc),
            )
            raise AirflowException(f"PostGameEvaluator.run() failed: {exc}") from exc

        slog.info(
            "post_game_eval_complete",
            game_date=game_date,
            games_resolved=metrics.games_resolved,
            mae=metrics.mae,
            rmse=metrics.rmse,
            mean_delta_er=metrics.mean_delta_er,
        )

        return {
            "game_date": metrics.game_date,
            "total_games": metrics.total_games,
            "games_resolved": metrics.games_resolved,
            "games_with_error": metrics.games_with_error,
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "mean_delta_er": metrics.mean_delta_er,
            "model_version": metrics.model_version,
        }

    # ===================================================================
    # Wiring — dependencias del DAG
    # ===================================================================
    games_list = fetch_todays_schedule()
    game_results = game_pipeline_dynamic_map.expand(game_info=games_list)
    resolve_task = await_and_resolve_all_games(games=games_list)

    # Task 3 depende explícitamente de TODAS las instancias del mapa
    game_results >> resolve_task


# ---------------------------------------------------------------------------
# Helpers de soporte (fuera del DAG)
# ---------------------------------------------------------------------------


def _build_projected_lineup(home_team_id: int, away_team_id: int) -> dict:
    """Construye un lineup proyectado usando el roster 40-Man del home team.

    Fallback cuando el lineup oficial no fue publicado antes del primer pitch.
    Toma los primeros 9 bateadores del roster activo, ordenados por tipo de posición.
    """
    try:
        data = _http_get_json(
            f"{_MLB_API_BASE}/teams/{home_team_id}/roster/40Man",
            params={"hydrate": "person(stats(type=career,group=hitting))"},
        )
        roster_entries = data.get("roster", [])

        # Priorizar posiciones ofensivas: C, 1B, 2B, 3B, SS, LF, CF, RF, DH
        position_priority = {"DH": 0, "RF": 1, "LF": 2, "CF": 3, "1B": 4,
                             "3B": 5, "SS": 6, "2B": 7, "C": 8}

        batters = []
        for entry in roster_entries:
            person = entry.get("person", {})
            position = entry.get("position", {}).get("abbreviation", "")
            if position in ("SP", "RP", "P"):
                continue
            pid = person.get("id")
            if pid:
                batters.append({
                    "player_id": int(pid),
                    "position": position,
                    "priority": position_priority.get(position, 9),
                })

        batters.sort(key=lambda x: x["priority"])
        projected_order = [b["player_id"] for b in batters[:9]]

        if len(projected_order) < 9:
            # Pad con ceros si el roster tiene menos de 9 posiciones ofensivas
            projected_order += [0] * (9 - len(projected_order))

    except Exception as exc:
        log.warning("Projected lineup fetch failed for team_id=%s: %s", home_team_id, exc)
        projected_order = [0] * 9

    # Para el pitcher rival (away team) usamos el probable pitcher
    away_pitcher_id = None
    try:
        pitcher_data = _http_get_json(
            f"{_MLB_API_BASE}/teams/{away_team_id}/roster/40Man",
            params={"season": str(datetime.now().year)},
        )
        for entry in pitcher_data.get("roster", []):
            if entry.get("position", {}).get("abbreviation") == "SP":
                away_pitcher_id = entry.get("person", {}).get("id")
                break
    except Exception as exc:
        log.warning("Away pitcher fetch failed for team_id=%s: %s", away_team_id, exc)

    return {
        "home_batting_order": projected_order,
        "away_batting_order": [0] * 9,
        "home_starting_pitcher_id": None,
        "away_starting_pitcher_id": away_pitcher_id,
    }


def _build_postgres_dsn() -> str:
    """Construye el DSN de psycopg2 desde la Airflow Connection 'mlb_predictions_db'.

    Fallback: variable de entorno MLB_POSTGRES_DSN si la Connection no existe.
    """
    try:
        pg_hook = PostgresHook(postgres_conn_id=_PG_CONN_ID)
        conn = pg_hook.get_connection(_PG_CONN_ID)
        return (
            f"host={conn.host} port={conn.port or 5432} "
            f"dbname={conn.schema} user={conn.login} password={conn.password}"
        )
    except Exception:
        dsn = os.environ.get("MLB_POSTGRES_DSN", "")
        if not dsn:
            raise RuntimeError(
                "PostgreSQL DSN no disponible. Configura la Airflow Connection "
                f"'{_PG_CONN_ID}' o la variable de entorno MLB_POSTGRES_DSN."
            )
        return dsn


# ---------------------------------------------------------------------------
# Instancia del DAG (requerido por Airflow para descubrir el DAG en el scheduler)
# ---------------------------------------------------------------------------

dag_instance = mlb_gameday_orchestrator()
