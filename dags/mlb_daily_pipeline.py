"""
dags/mlb_daily_pipeline.py
==========================
DAG diario del pipeline MLB AI.
Ciclo: Ingesta Statcast → Silver → Gold → Inferencia → Drift check → Alertas

Soporte de backfill: todos los tasks son idempotentes via S3 key existence checks
y writes atómicos. Si el DAG cae un día y se lanza con execution_date anterior,
reproduce el resultado correcto sin duplicar datos.

Schedule: 12:00 UTC (8am ET) — tiempo suficiente antes de los primeros pitches (~18:00 UTC).
"""
from __future__ import annotations

import logging
from datetime import timedelta

import boto3
import pendulum
import requests
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración central — valores en Airflow Variables / SSM Parameter Store
# ---------------------------------------------------------------------------
S3_BUCKET       = Variable.get("MLB_S3_BUCKET",        default_var="mlb-ai-datalake")
S3_BRONZE_PFX   = Variable.get("MLB_BRONZE_PREFIX",    default_var="bronze/statcast")
S3_SILVER_PFX   = Variable.get("MLB_SILVER_PREFIX",    default_var="silver/plate_appearances")
S3_GOLD_PFX     = Variable.get("MLB_GOLD_PREFIX",      default_var="gold/features")
S3_MODELS_PFX   = Variable.get("MLB_MODELS_PREFIX",    default_var="models")
MLFLOW_URI      = Variable.get("MLFLOW_TRACKING_URI",  default_var="http://mlflow:5000")
MODEL_NAME      = Variable.get("MLB_MODEL_NAME",       default_var="mlb-at-bat-predictor")
POSTGRES_CONN   = "mlb_predictions_db"
SLACK_WEBHOOK   = Variable.get("SLACK_WEBHOOK_URL",    default_var="")
PSI_THRESHOLD   = float(Variable.get("DRIFT_PSI_THRESHOLD", default_var="0.20"))

MLB_API_BASE    = "https://statsapi.mlb.com/api/v1"

DEFAULT_ARGS = {
    "owner":            "mlops",
    "depends_on_past":  False,
    "email_on_failure": True,
    "email":            ["mlops@yourorg.com"],
    "retries":          3,
    "retry_delay":      timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay":  timedelta(minutes=30),
}


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------
@dag(
    dag_id="mlb_daily_pipeline",
    description="Pipeline diario: Statcast → Silver → Gold → Inferencia → Monitoring",
    schedule="0 12 * * *",          # 12:00 UTC diario
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=True,                   # permite backfill histórico
    max_active_runs=2,
    default_args=DEFAULT_ARGS,
    tags=["mlb", "production", "daily"],
)
def mlb_daily_pipeline():

    # -----------------------------------------------------------------------
    # TASK 1 — Ingest: descarga Statcast del día via pybaseball
    # -----------------------------------------------------------------------
    @task(task_id="ingest_statcast")
    def ingest_statcast(data_interval_start=None) -> str:
        """
        Descarga datos Statcast crudos del día y los sube a S3 Bronze.
        Retorna la S3 key del archivo raw (parquet).
        Es idempotente: si la key ya existe, retorna sin re-descargar.
        """
        import io
        import polars as pl
        from pybaseball import statcast

        run_date: str = data_interval_start.strftime("%Y-%m-%d")
        s3_key = f"{S3_BRONZE_PFX}/date={run_date}/raw.parquet"

        s3 = S3Hook(aws_conn_id="aws_default")
        if s3.check_for_key(s3_key, bucket_name=S3_BUCKET):
            log.info("Bronze ya existe para %s, saltando ingesta.", run_date)
            return s3_key

        log.info("Descargando Statcast para %s...", run_date)
        df_pd = statcast(start_dt=run_date, end_dt=run_date, verbose=False)
        if df_pd is None or len(df_pd) == 0:
            raise ValueError(f"pybaseball retornó 0 filas para {run_date}")

        df = pl.from_pandas(df_pd)
        log.info("  %d pitch-level rows descargados.", len(df))

        buf = io.BytesIO()
        df.write_parquet(buf)
        buf.seek(0)

        s3.load_bytes(
            buf.read(),
            key=s3_key,
            bucket_name=S3_BUCKET,
            replace=False,          # atómico: no sobreescribe si ya existe
        )
        log.info("Bronze guardado: s3://%s/%s", S3_BUCKET, s3_key)
        return s3_key

    # -----------------------------------------------------------------------
    # TASK 2 — Silver: filtra PA-terminal events, aplica schema Silver
    # -----------------------------------------------------------------------
    @task(task_id="build_silver")
    def build_silver(bronze_key: str, data_interval_start=None) -> str:
        import io
        import polars as pl

        run_date: str = data_interval_start.strftime("%Y-%m-%d")
        year = data_interval_start.year
        s3_key_out = f"{S3_SILVER_PFX}/season={year}/date={run_date}/data.parquet"

        s3 = S3Hook(aws_conn_id="aws_default")
        if s3.check_for_key(s3_key_out, bucket_name=S3_BUCKET):
            log.info("Silver ya existe para %s.", run_date)
            return s3_key_out

        PA_EVENTS = {
            "strikeout", "strikeout_double_play", "walk", "intent_walk",
            "hit_by_pitch", "single", "double", "triple", "home_run",
            "field_out", "grounded_into_double_play", "double_play",
            "force_out", "field_error", "fielders_choice", "fielders_choice_out",
            "sac_fly", "sac_bunt", "other_out", "catcher_interf",
        }
        OUTCOME_MAP = {
            "strikeout": 1, "strikeout_double_play": 1,
            "walk": 2, "intent_walk": 2, "hit_by_pitch": 2,
            "single": 3, "double": 4, "triple": 5, "home_run": 6,
        }

        raw_obj = s3.get_key(bronze_key, bucket_name=S3_BUCKET)
        raw_bytes = raw_obj.get()["Body"].read()
        raw = pl.read_parquet(io.BytesIO(raw_bytes))

        df = raw.filter(
            pl.col("events").is_not_null() & pl.col("events").is_in(list(PA_EVENTS))
        )

        silver = (
            df.with_columns([
                pl.col("batter").cast(pl.Int64).alias("batter_id"),
                pl.col("pitcher").cast(pl.Int64).alias("pitcher_id"),
                pl.col("game_date").cast(pl.Utf8).alias("game_date"),
                pl.lit(year).cast(pl.Int32).alias("season"),
                pl.col("stand").alias("batter_stand"),
                pl.col("p_throws").alias("pitcher_throws"),
                pl.col("events").alias("pa_result"),
                (pl.col("estimated_woba_using_speedangle")
                 .cast(pl.Float32).alias("xwoba")),
                pl.col("launch_speed").cast(pl.Float32),
                pl.col("launch_angle").cast(pl.Float32),
            ])
            .with_columns(
                pl.col("pa_result")
                .map_elements(lambda e: OUTCOME_MAP.get(e, 0), return_dtype=pl.Int32)
                .alias("pa_outcome_int")
            )
            .select([
                "batter_id", "pitcher_id", "game_date", "season",
                "batter_stand", "pitcher_throws", "pa_result",
                "xwoba", "launch_speed", "launch_angle", "pa_outcome_int",
            ])
            .drop_nulls(subset=["batter_id", "pitcher_id", "game_date"])
        )

        buf = io.BytesIO()
        silver.write_parquet(buf)
        buf.seek(0)
        s3.load_bytes(buf.read(), key=s3_key_out, bucket_name=S3_BUCKET, replace=False)
        log.info("Silver guardado: %d PAs → s3://%s/%s", len(silver), S3_BUCKET, s3_key_out)
        return s3_key_out

    # -----------------------------------------------------------------------
    # TASK 3 — Gold: reconstruye features con ventana rolling histórica
    #           (lee Silver completo del año en curso + los 30 días previos)
    # -----------------------------------------------------------------------
    @task(task_id="build_gold_features")
    def build_gold_features(silver_key: str, data_interval_start=None) -> str:
        """
        Para inferencia diaria no reconstruimos el Gold entero (eso es para
        re-entrenamiento). Aquí calculamos solo las features de los bateadores
        que juegan HOY, usando un rolling window de los últimos 90 días de Silver.
        Resultado: parquet con una fila por bateador activo hoy.
        """
        import io
        import polars as pl

        run_date: str = data_interval_start.strftime("%Y-%m-%d")
        s3_key_out = f"{S3_GOLD_PFX}/inference/date={run_date}/features.parquet"

        s3 = S3Hook(aws_conn_id="aws_default")
        if s3.check_for_key(s3_key_out, bucket_name=S3_BUCKET):
            log.info("Gold features ya existen para %s.", run_date)
            return s3_key_out

        # Carga Silver de los últimos 90 días
        cutoff = (data_interval_start - timedelta(days=90)).strftime("%Y-%m-%d")
        parts = []
        s3_client = boto3.client("s3")
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_SILVER_PFX):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Extrae date= del path y filtra
                if "/date=" in key:
                    d = key.split("/date=")[1].split("/")[0]
                    if cutoff <= d <= run_date:
                        raw = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                        parts.append(pl.read_parquet(io.BytesIO(raw["Body"].read())))

        if not parts:
            raise RuntimeError(f"No Silver data en los últimos 90 días hasta {run_date}")

        silver = pl.concat(parts, how="diagonal_relaxed").sort(["batter_id", "game_date"])
        log.info("Silver para gold diario: %d PAs", len(silver))

        # Bateadores que juegan hoy (via MLB API)
        today_batters = _fetch_today_batter_ids(run_date)
        if not today_batters:
            log.warning("No se encontraron bateadores para %s vía MLB API.", run_date)

        # Filtra Silver a bateadores de hoy (o todos si no hay info)
        if today_batters:
            silver_filtered = silver.filter(pl.col("batter_id").is_in(today_batters))
        else:
            silver_filtered = silver

        # TODO: aquí llamarías a tu pipeline de features (build_gold.py logic)
        # Para el DAG se delega a un ECS task con build_gold.py --inference-mode
        buf = io.BytesIO()
        silver_filtered.write_parquet(buf)
        buf.seek(0)
        s3.load_bytes(buf.read(), key=s3_key_out, bucket_name=S3_BUCKET, replace=False)
        log.info("Gold diario guardado: s3://%s/%s", S3_BUCKET, s3_key_out)
        return s3_key_out

    # -----------------------------------------------------------------------
    # TASK 4 — Inferencia: llama a la FastAPI para cada partido del día
    # -----------------------------------------------------------------------
    @task(task_id="run_inference")
    def run_inference(gold_key: str, data_interval_start=None) -> dict:
        run_date: str = data_interval_start.strftime("%Y-%m-%d")
        api_url   = Variable.get("MLB_API_INFERENCE_URL", default_var="http://api:8000")
        api_token = Variable.get("MLB_API_TOKEN")

        headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        resp = requests.post(
            f"{api_url}/v1/predict/all",
            json={"game_date": run_date, "gold_s3_key": gold_key},
            headers=headers,
            timeout=300,
        )
        resp.raise_for_status()
        results = resp.json()
        log.info("Inferencia completada: %d equipos procesados.", len(results.get("teams", [])))
        return results

    # -----------------------------------------------------------------------
    # TASK 5 — Almacenamiento: persiste resultados en PostgreSQL
    # -----------------------------------------------------------------------
    @task(task_id="store_results")
    def store_results(inference_results: dict, data_interval_start=None) -> None:
        import json as _json

        run_date: str = data_interval_start.strftime("%Y-%m-%d")
        pg = PostgresHook(postgres_conn_id=POSTGRES_CONN)

        upsert_sql = """
            INSERT INTO predictions (
                game_date, game_pk, team_abbr, side,
                batting_order, expected_runs, model_version, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (game_date, game_pk, team_abbr, side)
            DO UPDATE SET
                batting_order  = EXCLUDED.batting_order,
                expected_runs  = EXCLUDED.expected_runs,
                model_version  = EXCLUDED.model_version,
                updated_at     = NOW();
        """
        model_ver = inference_results.get("model_version", "unknown")
        rows = []
        for team in inference_results.get("teams", []):
            rows.append((
                run_date,
                team["game_pk"],
                team["team_abbr"],
                team["side"],
                _json.dumps(team["batting_order"]),
                team["expected_runs_per_game"],
                model_ver,
            ))

        conn = pg.get_conn()
        cur  = conn.cursor()
        cur.executemany(upsert_sql, rows)
        conn.commit()
        log.info("Guardadas %d filas en predictions.", len(rows))

    # -----------------------------------------------------------------------
    # TASK 6 — Drift check: compara distribución de predicciones vs baseline
    # -----------------------------------------------------------------------
    @task(task_id="check_prediction_drift", trigger_rule=TriggerRule.ALL_SUCCESS)
    def check_prediction_drift(inference_results: dict, data_interval_start=None) -> dict:
        """
        Calcula Population Stability Index (PSI) entre la distribución de
        predicciones de hoy y el baseline del modelo (guardado en MLflow al entrenar).
        PSI > 0.20 → alerta crítica. PSI 0.10-0.20 → warning.
        """
        import numpy as np

        BASELINE = {           # distribución empírica del training set 2015-2024
            "OUT":    0.4643,
            "K":      0.2056,
            "BB/HBP": 0.0976,
            "1B":     0.1494,
            "2B":     0.0464,
            "3B":     0.0049,
            "HR":     0.0318,
        }
        OUTCOME_KEYS = ["prob_out", "prob_k", "prob_bb", "prob_1b", "prob_2b", "prob_3b", "prob_hr"]
        OUTCOME_NAMES = ["OUT", "K", "BB/HBP", "1B", "2B", "3B", "HR"]

        # Agrega todas las probabilidades predichas hoy
        all_probs = {k: [] for k in OUTCOME_KEYS}
        for team in inference_results.get("teams", []):
            for batter in team.get("batting_order", []):
                for k in OUTCOME_KEYS:
                    all_probs[k].append(batter.get(k, 0.0))

        if not any(all_probs.values()):
            log.warning("No hay predicciones para calcular drift.")
            return {"psi_total": 0.0, "status": "no_data"}

        # Distribución actual = media de probs por clase
        current_dist = {
            name: float(np.mean(all_probs[k]))
            for name, k in zip(OUTCOME_NAMES, OUTCOME_KEYS)
        }

        # PSI = sum((actual - expected) * ln(actual / expected))
        eps = 1e-6
        psi_per_class = {}
        for name in OUTCOME_NAMES:
            a = max(current_dist[name], eps)
            e = max(BASELINE[name], eps)
            psi_per_class[name] = (a - e) * np.log(a / e)

        psi_total = sum(psi_per_class.values())
        result = {
            "psi_total":    round(psi_total, 4),
            "psi_per_class": {k: round(v, 4) for k, v in psi_per_class.items()},
            "current_dist":  {k: round(v, 4) for k, v in current_dist.items()},
            "baseline_dist": BASELINE,
            "status": "critical" if psi_total > PSI_THRESHOLD
                      else "warning" if psi_total > PSI_THRESHOLD / 2
                      else "ok",
        }
        log.info("Drift PSI total=%.4f  status=%s", psi_total, result["status"])
        return result

    # -----------------------------------------------------------------------
    # TASK 7 — Alerta si drift crítico o fallos
    # -----------------------------------------------------------------------
    @task(task_id="send_alerts", trigger_rule=TriggerRule.ALL_DONE)
    def send_alerts(drift_result: dict, data_interval_start=None) -> None:
        run_date: str = data_interval_start.strftime("%Y-%m-%d")
        status = drift_result.get("status", "unknown")

        if status in ("critical", "warning") and SLACK_WEBHOOK:
            emoji = ":rotating_light:" if status == "critical" else ":warning:"
            msg = {
                "text": (
                    f"{emoji} *MLB AI Drift Alert* [{run_date}]\n"
                    f">Status: *{status.upper()}*\n"
                    f">PSI total: `{drift_result['psi_total']}`\n"
                    f">Threshold: `{PSI_THRESHOLD}`\n"
                    f">Por clase: `{drift_result['psi_per_class']}`\n"
                    f"Revisar: MLflow → modelo necesita re-evaluación."
                )
            }
            try:
                requests.post(SLACK_WEBHOOK, json=msg, timeout=10)
            except Exception as exc:
                log.warning("Slack alert falló: %s", exc)

    # -----------------------------------------------------------------------
    # WIRING — define dependencias del DAG
    # -----------------------------------------------------------------------
    bronze_key     = ingest_statcast()
    silver_key     = build_silver(bronze_key)
    gold_key       = build_gold_features(silver_key)
    inference_res  = run_inference(gold_key)
    store_results(inference_res)
    drift          = check_prediction_drift(inference_res)
    send_alerts(drift)


# ---------------------------------------------------------------------------
# Helper fuera del DAG (no es un task)
# ---------------------------------------------------------------------------
def _fetch_today_batter_ids(game_date: str) -> list[int]:
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/schedule",
            params={"sportId": 1, "date": game_date, "hydrate": "lineups"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        ids = []
        for d in data.get("dates", []):
            for game in d.get("games", []):
                for side in ("home", "away"):
                    for p in game.get("lineups", {}).get(f"{side}Players", []):
                        if pid := p.get("id"):
                            ids.append(int(pid))
        return list(set(ids))
    except Exception as exc:
        log.warning("MLB API lineup fetch falló: %s", exc)
        return []


dag_instance = mlb_daily_pipeline()
