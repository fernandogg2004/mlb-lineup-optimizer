"""
dags/mlb_gameday_orchestrator.py
=================================================================================================
Game-Day Orchestrator — Motor de operaciones en tiempo real del MLB Lineup Optimizer.

Programación: 08:00 UTC diario (10:00 Madrid / 04:00 EDT).

Flujo de 5 tareas con Dynamic Task Mapping:

    Task 1 — fetch_todays_schedule
        Consulta la MLB StatsAPI y devuelve la lista de partidos del día.
        Output: list[dict] — una entrada por gamePk.

    Task P — preliminary_optimization (×N, una por partido)
        Corre INMEDIATAMENTE después de Task 1. No espera el lineup oficial.
        Usa probable pitcher + roster proyectado (40-Man). fast_mode=True.
        Escribe gameday_predictions con lineup_source="preliminary".
        Streamlit puede mostrar predicciones desde las 10:00 AM sin esperar T-2h.

    Task 2a — wait_pregame_window (×N sensor, mode="reschedule")
        Sensor por partido que libera el worker slot mientras espera la ventana T-2h.
        Poke interval: 60 s. Timeout: 15 horas (partidos de Costa Oeste).

    Task 2b — lineup_sense_and_optimize (×N, una por partido)
        Corre tras wait_pregame_window[i] — NO antes.
        Llama al LineupSensor y, cuando el lineup oficial está disponible,
        invoca POST /v1/optimize/lineup con fast_mode=False (100k sims definitivas).
        UPSERT sobre la fila "preliminary" ya existente en gameday_predictions.
        Persiste confidence_score, confidence_level, scorecard_json, defense_report_md.

    Task 3 — await_and_resolve_all_games (trigger_rule=ALL_DONE)
        Corre cuando TODAS las Task 2b han terminado (éxito o fallo).
        PostGameEvaluator: espera estado Final → boxscores → delta E[R] → PDF → Slack.
        Al terminar, dispara TriggerDagRunOperator → mlb_nightly_training.

Zonas horarias:
    - Toda la lógica interna usa UTC (pytz.UTC).
    - La MLB StatsAPI devuelve gameDate en ISO 8601 UTC.
    - Conversiones a ET solo para logging y presentación.

Manejo de errores:
    - Task P: fallo individual no bloquea el pipeline. Retorna status="preliminary_failed".
    - Task 2a: timeout (15h) marca la tarea y la Task 2b correspondiente como failed.
    - Task 2b: retries=4, retry_delay=30min (polling del lineup oficial).
    - Task 3: trigger_rule=ALL_DONE — corre aunque alguna Task 2b haya fallado.
=================================================================================================
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any

import pendulum
import pytz
import requests
import structlog
from airflow.decorators import dag, task
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sensors.base import PokeReturnValue
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

# Intervalo de poke del sensor wait_pregame_window (segundos)
_WAIT_CHUNK_SECONDS = 60

# Intervalo entre polls del LineupSensor (cuando está en la ventana de 2 horas)
_LINEUP_POLL_SECONDS = 300  # 5 minutos (vía Airflow retries)

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
                raise
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
    raise RuntimeError(f"_http_get_json exhausted retries for {url}")


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
        except requests.exceptions.RequestException:
            if attempt == _HTTP_RETRIES:
                raise
            time.sleep(_HTTP_BACKOFF ** attempt)
    raise RuntimeError(f"_http_post_json exhausted retries for {url}")


# ---------------------------------------------------------------------------
# Helpers de lógica de negocio (sin estado de Airflow)
# ---------------------------------------------------------------------------


def _parse_game_dt_utc(game_date_str: str) -> datetime:
    """Convierte el gameDate de la MLB API (ISO 8601 UTC) en un datetime timezone-aware."""
    clean = game_date_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(clean)
    if dt.tzinfo is None:
        dt = _UTC.localize(dt)
    return dt.astimezone(_UTC)


def _fetch_schedule_for_date(target_date: str) -> list[dict]:
    """Obtiene la lista de partidos de la MLB para una fecha dada."""
    data = _http_get_json(
        f"{_MLB_API_BASE}/schedule",
        params={
            "sportId": _SPORT_ID,
            "date": target_date,
            "hydrate": "team,venue,status,probablePitcher",
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

            # Extract probable pitchers from hydrated schedule data
            teams_data = game.get("teams", {})
            home_probable = teams_data.get("home", {}).get("probablePitcher", {})
            away_probable = teams_data.get("away", {}).get("probablePitcher", {})

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
                # Probable pitchers (available since we now hydrate probablePitcher)
                "home_probable_pitcher_id": home_probable.get("id"),
                "away_probable_pitcher_id": away_probable.get("id"),
            })

    log.info("Schedule fetched for %s: %d games found.", target_date, len(games))
    return games


def _sense_lineup(game_pk: int) -> tuple[bool, str, dict]:
    """Consulta el live-feed de un partido y detecta si el lineup oficial está disponible."""
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
    """Obtiene estadísticas de carrera de un bateador desde la MLB Stats API."""
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

    Fallback cuando el feature store no está disponible en el worker de Airflow.
    En producción el prob_vector debe calcularse con el modelo AtBatPredictor.
    """
    avg = float(stats.get("avg", 0.248))
    obp = float(stats.get("obp", 0.318))
    slg = float(stats.get("slg", 0.398))

    pa_adj = 0.97
    p_bb_hbp = max(round((obp - avg) * pa_adj, 4), 0.04)
    p_hr = max(round((slg - avg) / 4.2, 4), 0.01)
    p_triple = 0.005
    p_double = max(round((slg - avg - p_hr * 3 - p_triple * 2) * 0.35, 4), 0.02)
    p_single = max(round(avg - p_hr - p_triple - p_double, 4), 0.08)
    p_k = max(round(0.22 - (obp - 0.310) * 0.6, 4), 0.08)
    p_out = max(round(1.0 - p_bb_hbp - p_hr - p_triple - p_double - p_single - p_k, 4), 0.20)

    raw = [p_out, p_k, p_bb_hbp, p_single, p_double, p_triple, p_hr]
    total = sum(raw)
    return [round(x / total, 4) for x in raw]


def _build_optimize_request(
    game_info: dict,
    lineup_data: dict,
    fast_mode: bool = True,
) -> dict | None:
    """Construye el payload para POST /v1/optimize/lineup.

    Args:
        game_info:   Dict producido por fetch_todays_schedule.
        lineup_data: Dict con home_batting_order, away_batting_order,
                     home_starting_pitcher_id, away_starting_pitcher_id.
        fast_mode:   True → 5k sims (preliminary/fallback).
                     False → 100k sims definitivas (lineup oficial confirmado).

    Returns:
        Payload dict compatible con LineupOptimizeRequest, o None si no se
        pudo construir el roster completo (< 9 jugadores con stats).
    """
    home_order = lineup_data.get("home_batting_order", [])
    away_pitcher_id = lineup_data.get("away_starting_pitcher_id")

    if len(home_order) < 9:
        log.warning(
            "build_optimize_request: game_pk=%s — home batting order has only %d players",
            game_info["game_pk"], len(home_order),
        )
        return None

    roster: list[dict] = []
    for pid in home_order[:9]:
        stats = _fetch_player_career_stats(int(pid))
        if stats is None:
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

    rival_pitcher: dict = {
        "pitcher_id": away_pitcher_id or 0,
        "pitcher_name": f"Pitcher {away_pitcher_id}",
        "hand": "R",
        "era": 4.00,
        "pitch_mix": {"FF": 0.55, "SL": 0.20, "CH": 0.15, "CB": 0.10},
    }

    if away_pitcher_id:
        try:
            pitcher_data = _http_get_json(
                f"{_MLB_API_BASE}/people/{away_pitcher_id}",
                params={"hydrate": "stats(group=[pitching],type=[season])"},
            )
            people = pitcher_data.get("people", [])
            if people:
                p = people[0]
                rival_pitcher["pitcher_name"] = p.get("fullName", rival_pitcher["pitcher_name"])
                rival_pitcher["hand"] = p.get("pitchHand", {}).get("code", "R")
                # Extract ERA from season stats if available
                for s in p.get("stats", []):
                    if s.get("type", {}).get("displayName") == "season":
                        splits = s.get("splits", [])
                        if splits:
                            stat = splits[0].get("stat", {})
                            era_str = stat.get("era", "4.00")
                            try:
                                rival_pitcher["era"] = float(era_str)
                            except (ValueError, TypeError):
                                pass
                        break
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
        "fast_mode": fast_mode,
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

    Columnas adicionales (requiere migration 002_add_confidence_columns.sql):
        confidence_score, confidence_level, scorecard_json, defense_report_md

    La estrategia UPSERT usa COALESCE para los nuevos campos: si la fila existe
    con un valor no-NULL (ej. scorecard de la predicción definitiva), no se
    sobreescribe con NULL (ej. actualización parcial de Task 3 post-partido).
    """
    # ── Extraer scorecard y defense_report de la respuesta de la API ──────────
    scorecard: dict | None = prediction.get("scorecard")
    defense_report: dict | None = prediction.get("defense_report")

    confidence_score: float | None = scorecard.get("confidence_overall") if scorecard else None
    confidence_level: str | None = scorecard.get("confidence_tier") if scorecard else None
    scorecard_json: str | None = json.dumps(scorecard) if scorecard else None
    defense_report_md: str | None = (
        defense_report.get("markdown_content") if defense_report else None
    )

    upsert_sql = """
        INSERT INTO gameday_predictions (
            game_pk, game_date,
            home_team_id, away_team_id,
            home_team_name, away_team_name,
            home_batting_order, away_batting_order,
            home_starting_pitcher_id, away_starting_pitcher_id,
            ai_recommended_lineup, ai_expected_runs,
            win_probability, optimization_mode,
            model_version, lineup_source, predicted_at,
            confidence_score, confidence_level, scorecard_json, defense_report_md
        ) VALUES (
            %(game_pk)s, %(game_date)s,
            %(home_team_id)s, %(away_team_id)s,
            %(home_team_name)s, %(away_team_name)s,
            %(home_batting_order)s, %(away_batting_order)s,
            %(home_starting_pitcher_id)s, %(away_starting_pitcher_id)s,
            %(ai_recommended_lineup)s, %(ai_expected_runs)s,
            %(win_probability)s, %(optimization_mode)s,
            %(model_version)s, %(lineup_source)s, NOW(),
            %(confidence_score)s, %(confidence_level)s,
            %(scorecard_json)s::jsonb, %(defense_report_md)s
        )
        ON CONFLICT (game_pk) DO UPDATE SET
            ai_recommended_lineup    = EXCLUDED.ai_recommended_lineup,
            ai_expected_runs         = EXCLUDED.ai_expected_runs,
            win_probability          = EXCLUDED.win_probability,
            optimization_mode        = EXCLUDED.optimization_mode,
            model_version            = EXCLUDED.model_version,
            lineup_source            = EXCLUDED.lineup_source,
            predicted_at             = NOW(),
            home_batting_order       = EXCLUDED.home_batting_order,
            away_batting_order       = EXCLUDED.away_batting_order,
            home_starting_pitcher_id = EXCLUDED.home_starting_pitcher_id,
            away_starting_pitcher_id = EXCLUDED.away_starting_pitcher_id,
            -- COALESCE: preserve existing non-NULL scorecard/defense_report when
            -- updating with a preliminary prediction that has NULL fields.
            confidence_score  = COALESCE(EXCLUDED.confidence_score,  gameday_predictions.confidence_score),
            confidence_level  = COALESCE(EXCLUDED.confidence_level,  gameday_predictions.confidence_level),
            scorecard_json    = COALESCE(EXCLUDED.scorecard_json,    gameday_predictions.scorecard_json),
            defense_report_md = COALESCE(EXCLUDED.defense_report_md, gameday_predictions.defense_report_md)
    """
    params = {
        "game_pk":                   game_info["game_pk"],
        "game_date":                 game_info["game_date_utc"][:10],
        "home_team_id":              game_info["home_team_id"],
        "away_team_id":              game_info["away_team_id"],
        "home_team_name":            game_info["home_team_name"],
        "away_team_name":            game_info["away_team_name"],
        "home_batting_order":        lineup_data.get("home_batting_order", []),
        "away_batting_order":        lineup_data.get("away_batting_order", []),
        "home_starting_pitcher_id":  lineup_data.get("home_starting_pitcher_id"),
        "away_starting_pitcher_id":  lineup_data.get("away_starting_pitcher_id"),
        "ai_recommended_lineup":     prediction.get("lineup_player_ids", []),
        "ai_expected_runs":          prediction.get("expected_runs", 0.0),
        "win_probability":           prediction.get("win_probability", 0.0),
        "optimization_mode":         prediction.get("optimization_mode", "fast"),
        "model_version":             prediction.get("model_version", "unknown"),
        "lineup_source":             lineup_source,
        "confidence_score":          confidence_score,
        "confidence_level":          confidence_level,
        "scorecard_json":            scorecard_json,
        "defense_report_md":         defense_report_md,
    }
    conn = pg_hook.get_conn()
    cur = conn.cursor()
    cur.execute(upsert_sql, params)
    conn.commit()
    slog.info(
        "prediction_saved_to_db",
        game_pk=game_info["game_pk"],
        lineup_source=lineup_source,
        ai_expected_runs=params["ai_expected_runs"],
        confidence_level=confidence_level,
        has_defense_report=defense_report_md is not None,
    )


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------


@dag(
    dag_id="mlb_gameday_orchestrator",
    description=(
        "Game-Day Orchestrator v2: preliminary sim (08:00 UTC) + "
        "DateTimeSensor (T-2h, mode=reschedule) + definitive optimization "
        "(100k sims) + post-game eval + TriggerDagRun → mlb_nightly_training."
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
    """DAG del Game-Day Orchestrator v2 del MLB Lineup Optimizer."""

    # ===================================================================
    # TASK 1 — Fetch Today's Schedule
    # ===================================================================

    @task(task_id="fetch_todays_schedule")
    def fetch_todays_schedule(**context) -> list[dict]:
        """Obtiene los partidos MLB para hoy, incluyendo probable pitchers.

        Usa la logical_date del DAG run para garantizar idempotencia en backfills.
        El hydrate=probablePitcher permite que preliminary_optimization tenga
        el probable pitcher sin llamadas adicionales.
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
    # TASK P — Preliminary Optimization (corre inmediatamente, sin esperar T-2h)
    # ===================================================================

    @task(
        task_id="preliminary_optimization",
        retries=1,
        retry_delay=timedelta(minutes=3),
        execution_timeout=timedelta(minutes=10),
    )
    def preliminary_optimization(game_info: dict, **context) -> dict:
        """Simulación preliminar en fast_mode usando probable pitcher + roster proyectado.

        Escribe gameday_predictions con lineup_source="preliminary" a las 08:00 UTC,
        permitiendo que Streamlit War Room muestre predicciones antes de T-2h.

        Los fallos individuales no bloquean el pipeline — se loguean y se retornan
        como status="preliminary_failed" para que Task 3 los detecte.
        """
        game_pk: int = game_info["game_pk"]
        matchup = f"{game_info['away_team_name']} @ {game_info['home_team_name']}"

        slog.info("preliminary_optimization_start", game_pk=game_pk, matchup=matchup)

        # Construir lineup proyectado (40-Man roster del home team)
        lineup_data = _build_projected_lineup(
            game_info["home_team_id"],
            game_info["away_team_id"],
        )

        # Usar el probable pitcher (ya disponible en game_info desde fetch_schedule)
        away_probable_id = game_info.get("away_probable_pitcher_id")
        if away_probable_id:
            lineup_data["away_starting_pitcher_id"] = int(away_probable_id)

        optimize_payload = _build_optimize_request(game_info, lineup_data, fast_mode=True)
        if optimize_payload is None:
            slog.warning("preliminary_roster_build_failed", game_pk=game_pk, matchup=matchup)
            return {
                "game_pk": game_pk,
                "matchup": matchup,
                "lineup_source": "preliminary",
                "ai_expected_runs": None,
                "status": "preliminary_skipped_no_roster",
            }

        api_url = Variable.get("MLB_API_INFERENCE_URL", default_var="http://mlb_api:8000")
        try:
            prediction = _http_post_json(
                f"{api_url}/v1/optimize/lineup",
                payload=optimize_payload,
            )
        except Exception as exc:
            slog.warning(
                "preliminary_inference_failed",
                game_pk=game_pk,
                matchup=matchup,
                error=str(exc),
            )
            # Non-fatal: log + return. Official prediction at T-2h is the definitive one.
            return {
                "game_pk": game_pk,
                "matchup": matchup,
                "lineup_source": "preliminary",
                "ai_expected_runs": None,
                "status": f"preliminary_inference_failed: {exc}",
            }

        pg_hook = PostgresHook(postgres_conn_id=_PG_CONN_ID)
        try:
            _save_prediction_to_db(pg_hook, game_info, lineup_data, prediction, "preliminary")
        except Exception as exc:
            slog.error("preliminary_db_save_failed", game_pk=game_pk, error=str(exc))
            return {
                "game_pk": game_pk,
                "matchup": matchup,
                "lineup_source": "preliminary",
                "ai_expected_runs": prediction.get("expected_runs"),
                "status": f"preliminary_db_failed: {exc}",
            }

        slog.info(
            "preliminary_optimization_complete",
            game_pk=game_pk,
            matchup=matchup,
            ai_expected_runs=prediction.get("expected_runs"),
            win_probability=prediction.get("win_probability"),
        )
        return {
            "game_pk": game_pk,
            "matchup": matchup,
            "lineup_source": "preliminary",
            "ai_expected_runs": prediction.get("expected_runs"),
            "win_probability": prediction.get("win_probability"),
            "status": "ok",
        }

    # ===================================================================
    # TASK 2a — Wait Pregame Window (sensor, mode="reschedule")
    # ===================================================================

    @task.sensor(
        task_id="wait_pregame_window",
        mode="reschedule",        # libera el worker slot entre pokes
        poke_interval=_WAIT_CHUNK_SECONDS,
        timeout=int(timedelta(hours=15).total_seconds()),  # partidos de Costa Oeste
        soft_fail=False,
    )
    def wait_pregame_window(game_info: dict) -> PokeReturnValue:
        """Sensor que espera la ventana T-2h antes del primer pitch.

        Usa mode="reschedule" para NO mantener el worker slot ocupado durante
        la espera (que puede ser de hasta 12 horas para partidos de Costa Oeste).
        El worker se libera entre pokes y solo se reactiva cada 60 segundos.

        PokeReturnValue(is_done=True) hace avanzar la Task 2b correspondiente.
        """
        game_pk = game_info["game_pk"]
        game_dt_utc = _parse_game_dt_utc(game_info["game_date_utc"])
        window_start_utc = game_dt_utc - timedelta(hours=_LINEUP_WINDOW_HOURS)
        now_utc = datetime.now(_UTC)

        is_ready = now_utc >= window_start_utc

        if not is_ready:
            remaining_s = (window_start_utc - now_utc).total_seconds()
            slog.debug(
                "pregame_window_not_yet",
                game_pk=game_pk,
                window_start_utc=window_start_utc.strftime("%Y-%m-%d %H:%M UTC"),
                remaining_minutes=round(remaining_s / 60, 1),
            )
        else:
            slog.info(
                "pregame_window_reached",
                game_pk=game_pk,
                matchup=f"{game_info['away_team_name']} @ {game_info['home_team_name']}",
            )

        return PokeReturnValue(is_done=is_ready)

    # ===================================================================
    # TASK 2b — Lineup Sense and Optimize (corre después de wait_pregame_window[i])
    # ===================================================================

    @task(
        task_id="lineup_sense_and_optimize",
        retries=4,
        retry_delay=timedelta(minutes=30),
        execution_timeout=timedelta(hours=2),
    )
    def lineup_sense_and_optimize(game_info: dict, **context) -> dict:
        """Sensing del lineup oficial + optimización definitiva para un único gamePk.

        PRECONDICIÓN: wait_pregame_window[i] completó — ya estamos en la ventana T-2h.

        Etapas:
            1. LineupSensor: consultar lineup oficial.
               → Si disponible: fast_mode=False (100k sims — decisión definitiva).
               → Si partido ya inició sin lineup: roster proyectado, fast_mode=True.
               → Si no disponible y no inició: raise AirflowException → retry 30min.
            2. POST /v1/optimize/lineup con los params correctos.
            3. UPSERT en gameday_predictions (sobrescribe el registro "preliminary").
               Persiste confidence_score, confidence_level, scorecard_json,
               defense_report_md del modo full (100k sims).
        """
        game_pk: int = game_info["game_pk"]
        matchup = f"{game_info['away_team_name']} @ {game_info['home_team_name']}"

        slog.info(
            "lineup_sense_start",
            game_pk=game_pk,
            matchup=matchup,
            map_index=context.get("map_index_template", "?"),
        )

        # ------------------------------------------------------------------
        # Etapa 1: LineupSensor (ya estamos en la ventana T-2h)
        # ------------------------------------------------------------------
        lineup_available, game_state, lineup_data = _sense_lineup(game_pk)

        fast_mode: bool
        lineup_source: str

        if lineup_available:
            lineup_source = "official"
            fast_mode = False  # Decisión definitiva: 100k simulaciones
            slog.info(
                "lineup_confirmed",
                game_pk=game_pk,
                matchup=matchup,
                source=lineup_source,
                home_order=lineup_data["home_batting_order"],
                home_starter=lineup_data["home_starting_pitcher_id"],
            )

        elif game_state in ("Live", "In Progress", "Final", "Game Over"):
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
            fast_mode = True  # Game already started: best-effort fast estimate

        else:
            slog.warning(
                "lineup_not_available_triggering_retry",
                game_pk=game_pk,
                game_state=game_state,
                matchup=matchup,
                retries_remaining=(
                    context["task_instance"].max_tries - context["task_instance"].try_number
                ),
            )
            raise AirflowException(
                f"game_pk={game_pk} ({matchup}): lineup not published yet "
                f"(state={game_state}). Retry in 30 min."
            )

        # ------------------------------------------------------------------
        # Etapa 2: Optimización
        # ------------------------------------------------------------------
        optimize_payload = _build_optimize_request(game_info, lineup_data, fast_mode=fast_mode)
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

        api_url = Variable.get("MLB_API_INFERENCE_URL", default_var="http://mlb_api:8000")

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
            fast_mode=fast_mode,
            ai_expected_runs=prediction.get("expected_runs"),
            win_probability=prediction.get("win_probability"),
            confidence_level=prediction.get("scorecard", {}) and
                             prediction["scorecard"].get("confidence_tier"),
            has_defense_report=prediction.get("defense_report") is not None,
        )

        # ------------------------------------------------------------------
        # Etapa 3: UPSERT en PostgreSQL
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
            raise AirflowException(f"game_pk={game_pk}: DB save failed: {exc}") from exc

        return {
            "game_pk": game_pk,
            "matchup": matchup,
            "lineup_source": lineup_source,
            "fast_mode": fast_mode,
            "ai_expected_runs": prediction.get("expected_runs"),
            "win_probability": prediction.get("win_probability"),
            "model_version": prediction.get("model_version"),
            "confidence_level": (
                prediction.get("scorecard", {}) and
                prediction["scorecard"].get("confidence_tier")
            ),
            "status": "ok",
        }

    # ===================================================================
    # TASK 3 — Post-game Evaluation (Join dinámico)
    # ===================================================================

    @task(
        task_id="await_and_resolve_all_games",
        trigger_rule=TriggerRule.ALL_DONE,
        retries=2,
        retry_delay=timedelta(minutes=15),
        execution_timeout=timedelta(hours=10),
    )
    def await_and_resolve_all_games(games: list[dict], **context) -> dict:
        """Evaluación analítica post-partido de todos los juegos del día.

        Se ejecuta cuando TODAS las instancias de lineup_sense_and_optimize
        han terminado (éxito o fallo). Usa PostGameEvaluator para:
            - Esperar que todos los partidos lleguen a estado Final.
            - Descargar boxscores y cruzar con predicciones en DB.
            - Calcular delta E[R], MAE, RMSE.
            - Generar PDF con reportlab.
            - Guardar resoluciones en PostgreSQL.
            - Notificar por Slack.

        El TriggerDagRunOperator hacia mlb_nightly_training se define en el
        wiring del DAG (fuera de esta task) y se ejecuta tras su completion.
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
            "game_date":        metrics.game_date,
            "total_games":      metrics.total_games,
            "games_resolved":   metrics.games_resolved,
            "games_with_error": metrics.games_with_error,
            "mae":              metrics.mae,
            "rmse":             metrics.rmse,
            "mean_delta_er":    metrics.mean_delta_er,
            "model_version":    metrics.model_version,
        }

    # ===================================================================
    # Wiring — dependencias del DAG
    # ===================================================================
    games_list = fetch_todays_schedule()

    # Task P: simulación preliminar inmediata (depende solo de Task 1)
    preliminary_results = preliminary_optimization.expand(game_info=games_list)

    # Task 2a: sensor por partido (libera workers mientras espera T-2h)
    wait_sensors = wait_pregame_window.expand(game_info=games_list)

    # Task 2b: sense + optimize definitivo — cada [i] depende de wait_sensors[i]
    official_results = lineup_sense_and_optimize.expand(game_info=games_list)
    wait_sensors >> official_results

    # Task 3: post-game eval — espera que TODOS los preliminary Y official terminen
    resolve_task = await_and_resolve_all_games(games=games_list)
    [preliminary_results, official_results] >> resolve_task

    # Trigger del pipeline de reentrenamiento nocturno al finalizar la evaluación
    trigger_training = TriggerDagRunOperator(
        task_id="trigger_nightly_training",
        trigger_dag_id="mlb_nightly_training",
        conf={"game_date": "{{ ds }}"},
        wait_for_completion=False,  # No bloquear el DAG esperando el entrenamiento
        reset_dag_run=False,
        poke_interval=30,
    )
    resolve_task >> trigger_training


# ---------------------------------------------------------------------------
# Helpers de soporte (fuera del DAG)
# ---------------------------------------------------------------------------


def _build_projected_lineup(home_team_id: int, away_team_id: int) -> dict:
    """Construye un lineup proyectado usando el roster 40-Man del home team.

    Fallback cuando el lineup oficial no fue publicado antes del primer pitch,
    o para la predicción preliminar de las 08:00 UTC.
    """
    try:
        data = _http_get_json(
            f"{_MLB_API_BASE}/teams/{home_team_id}/roster/40Man",
            params={"hydrate": "person(stats(type=career,group=hitting))"},
        )
        roster_entries = data.get("roster", [])

        position_priority = {
            "DH": 0, "RF": 1, "LF": 2, "CF": 3,
            "1B": 4, "3B": 5, "SS": 6, "2B": 7, "C": 8,
        }

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
            projected_order += [0] * (9 - len(projected_order))

    except Exception as exc:
        log.warning("Projected lineup fetch failed for team_id=%s: %s", home_team_id, exc)
        projected_order = [0] * 9

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
        "home_batting_order":        projected_order,
        "away_batting_order":        [0] * 9,
        "home_starting_pitcher_id":  None,
        "away_starting_pitcher_id":  away_pitcher_id,
    }


def _build_postgres_dsn() -> str:
    """Construye el DSN de psycopg2 desde la Airflow Connection 'mlb_predictions_db'."""
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
# Instancia del DAG (requerido por Airflow para el scheduler)
# ---------------------------------------------------------------------------

dag_instance = mlb_gameday_orchestrator()
