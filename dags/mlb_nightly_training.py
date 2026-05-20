"""
dags/mlb_nightly_training.py
=================================================================================================
Nightly MLOps Pipeline — Reentrenamiento continuo del MLB Lineup Optimizer.

Disparado por mlb_gameday_orchestrator vía TriggerDagRunOperator al finalizar
la evaluación post-partido. También puede ejecutarse manualmente para backfills:

    airflow dags trigger mlb_nightly_training --conf '{"game_date": "2026-05-19"}'

Flujo de tareas:

    ┌── close_experiment_observations ──────────────────────────────────────────┐
    │   Cierra observaciones A/B abiertas via PostGameResolver.                  │
    │   Independiente: corre en paralelo con ingest_statcast_yesterday.          │
    └────────────────────────────────────────────────────────────────────────────┘

    ┌── ingest_statcast_yesterday ──→ retrain_challenger ──→ auto_promote ───────┐
    │   Descarga datos Statcast     │  GameDayRetraining-   │  AutoPromoter gate  │
    │   del día anterior via        │  Pipeline.run():       │  Δlog-loss + ECE.  │
    │   pybaseball → Gold Parquet.  │  features + train +    │  ECS deploy si     │
    │   Agrega a PA-level.          │  MLflow staging alias. │  ambos gates pasan.│
    └───────────────────────────────┴────────────────────────┴────────────────────┘

Separación de concerns (Opción B):
    mlb_gameday_orchestrator  lifecycle ~12h  (08:00 UTC → ~20:00 UTC)
    mlb_nightly_training      lifecycle ~3h   (disparado tras el último out)

    Aislamiento de fallos: si Statcast API falla, el reentrenamiento falla en
    su propio DAG. El pipeline proactivo del día siguiente NO se bloquea y
    el modelo Champion actual sigue en producción.
=================================================================================================
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pendulum
import structlog
from airflow.decorators import dag, task
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule

slog = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Mapeo de eventos Statcast → clase de resultado de turno al bate (7 clases)
# Debe ser consistente con el entrenamiento del AtBatPredictor.
# ---------------------------------------------------------------------------
_EVENT_TO_PA_OUTCOME: dict[str, int] = {
    # 0 — OUT_IN_PLAY
    "field_out": 0,
    "grounded_into_double_play": 0,
    "fielders_choice": 0,
    "fielders_choice_out": 0,
    "force_out": 0,
    "sac_fly": 0,
    "sac_bunt": 0,
    "sac_fly_double_play": 0,
    "double_play": 0,
    "triple_play": 0,
    "other_out": 0,
    # 1 — STRIKEOUT
    "strikeout": 1,
    "strikeout_double_play": 1,
    # 2 — WALK_HBP
    "walk": 2,
    "hit_by_pitch": 2,
    "catcher_interf": 2,
    "intent_walk": 2,
    # 3 — SINGLE
    "single": 3,
    # 4 — DOUBLE
    "double": 4,
    # 5 — TRIPLE
    "triple": 5,
    # 6 — HOME_RUN
    "home_run": 6,
}

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "mlb-mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": Variable.get("ALERT_EMAIL", default_var="mlops@yourorg.com"),
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


# ---------------------------------------------------------------------------
# Helper: Statcast pitch-level → plate appearance level
# ---------------------------------------------------------------------------


def _statcast_to_plate_appearances(df_raw: Any, target_date: date) -> Any:
    """Agrega datos de pitch-level de Statcast a plate-appearance level.

    Filtra a las filas donde `events` es no-nulo (pitch final de un PA),
    mapea el evento a `pa_outcome_int` (7 clases) y proyecta las columnas
    mínimas necesarias por FeatureMatrixBuilder.

    Args:
        df_raw:      DataFrame de pandas con datos raw de pybaseball.statcast().
        target_date: Fecha del partido (para `game_date` y `season`).

    Returns:
        DataFrame de pandas con columnas:
        batter_id, pitcher_id, game_date, season, pa_outcome_int, pa_id,
        stand (batter_stand), p_throws (pitcher_hand), game_pk.
    """
    import pandas as pd

    # Solo filas que terminan un PA (event no nulo)
    df = df_raw[df_raw["events"].notna()].copy()

    if df.empty:
        return df

    # Mapear evento → clase numérica
    df["pa_outcome_int"] = df["events"].map(_EVENT_TO_PA_OUTCOME)

    # Descartar eventos sin mapeo definido (e.g., "caught_stealing_2b")
    df = df[df["pa_outcome_int"].notna()].copy()
    df["pa_outcome_int"] = df["pa_outcome_int"].astype(int)

    # Identificador único de PA: game_pk + at_bat_number
    df["pa_id"] = (
        df["game_pk"].astype(str) + "_" + df["at_bat_number"].astype(str)
    )

    df["game_date"] = target_date.isoformat()
    df["season"] = target_date.year

    # Proyección de columnas relevantes
    cols_available = {
        "batter":       "batter_id",
        "pitcher":      "pitcher_id",
        "game_pk":      "game_pk",
        "stand":        "batter_stand",
        "p_throws":     "pitcher_hand",
    }
    rename_map = {k: v for k, v in cols_available.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    keep_cols = [
        "batter_id", "pitcher_id", "game_pk", "game_date", "season",
        "pa_outcome_int", "pa_id", "batter_stand", "pitcher_hand",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset=["pa_id"])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------


@dag(
    dag_id="mlb_nightly_training",
    description=(
        "Nightly MLOps: A/B experiment closure + Statcast ingest + "
        "challenger retrain (LightGBM) + AutoPromoter gate → ECS deploy."
    ),
    schedule=None,  # Triggered exclusively by mlb_gameday_orchestrator
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlb", "mlops", "training", "nightly"],
    doc_md=__doc__,
)
def mlb_nightly_training():
    """Pipeline nocturno de reentrenamiento continuo."""

    def _get_game_date(context: dict) -> str:
        """Extrae game_date del conf del dag_run, con fallback al logical_date."""
        conf = context.get("dag_run", {}).conf or {}
        return conf.get("game_date", context["ds"])

    # ===================================================================
    # TASK A — Cierre de Observaciones A/B (independiente)
    # ===================================================================

    @task(
        task_id="close_experiment_observations",
        retries=2,
        retry_delay=timedelta(minutes=5),
        execution_timeout=timedelta(minutes=30),
    )
    def close_experiment_observations(**context) -> dict:
        """Cierra observaciones A/B abiertas usando PostGameResolver.

        Escanea los últimos RESOLVER_LOOKBACK_DAYS días de observaciones
        con status="open" y descarga los boxscores reales para cerrarlas.
        Corre en paralelo con ingest_statcast_yesterday.
        """
        from src.experiment.post_game_resolver import PostGameResolver

        game_date = _get_game_date(context)
        slog.info("experiment_closure_start", game_date=game_date)

        resolver = PostGameResolver.from_env()

        try:
            summaries = resolver.resolve_pending_outcomes(target_date=game_date)
        except Exception as exc:
            slog.error("experiment_closure_failed", game_date=game_date, error=str(exc))
            raise AirflowException(f"PostGameResolver failed for {game_date}: {exc}") from exc

        total_resolved = sum(s.resolved for s in summaries)
        total_errors = sum(s.errors for s in summaries)
        total_open = sum(s.open_found for s in summaries)

        slog.info(
            "experiment_closure_complete",
            game_date=game_date,
            dates_checked=len(summaries),
            open_found=total_open,
            resolved=total_resolved,
            errors=total_errors,
        )

        if total_errors > 0:
            slog.warning(
                "experiment_closure_partial_errors",
                errors=total_errors,
                resolved=total_resolved,
            )

        return {
            "game_date":     game_date,
            "dates_checked": len(summaries),
            "open_found":    total_open,
            "resolved":      total_resolved,
            "errors":        total_errors,
        }

    # ===================================================================
    # TASK B — Ingesta Statcast (inicio de la cadena de entrenamiento)
    # ===================================================================

    @task(
        task_id="ingest_statcast_yesterday",
        retries=2,
        retry_delay=timedelta(minutes=15),
        execution_timeout=timedelta(minutes=45),
    )
    def ingest_statcast_yesterday(**context) -> dict:
        """Descarga datos Statcast del día anterior via pybaseball.

        Agrega pitch-level → plate-appearance level y escribe a
        data/gold/plate_appearances/game_date=YYYY-MM-DD/part-0.parquet.

        Partición hive compatible con FeatureMatrixBuilder de
        GameDayRetrainingPipeline.
        """
        import pandas as pd

        game_date_str = _get_game_date(context)
        target_date = date.fromisoformat(game_date_str)

        try:
            import pybaseball as pyb
            pyb.cache.enable()
        except ImportError as exc:
            raise AirflowException(
                "pybaseball not installed. Add it to requirements-airflow.txt."
            ) from exc

        slog.info("statcast_ingest_start", game_date=game_date_str)

        try:
            df_raw = pyb.statcast(
                start_dt=game_date_str,
                end_dt=game_date_str,
                verbose=False,
            )
        except Exception as exc:
            slog.error("statcast_download_failed", game_date=game_date_str, error=str(exc))
            raise AirflowException(
                f"Statcast download failed for {game_date_str}: {exc}"
            ) from exc

        if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
            slog.info("statcast_no_data", game_date=game_date_str)
            raise AirflowSkipException(f"No Statcast data available for {game_date_str}.")

        slog.info(
            "statcast_raw_downloaded",
            game_date=game_date_str,
            raw_pitches=len(df_raw),
        )

        df_pa = _statcast_to_plate_appearances(df_raw, target_date)

        if df_pa.empty:
            slog.warning("statcast_no_valid_pas", game_date=game_date_str)
            raise AirflowSkipException(
                f"No valid plate appearance events in Statcast data for {game_date_str}."
            )

        # Write to Gold Parquet (hive partition)
        gold_pa_root = Path(
            os.environ.get("GOLD_PA_PATH", "data/gold/plate_appearances")
        )
        partition_dir = gold_pa_root / f"game_date={game_date_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        output_path = partition_dir / "part-0.parquet"

        df_pa.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")

        slog.info(
            "statcast_ingest_complete",
            game_date=game_date_str,
            plate_appearances=len(df_pa),
            output_path=str(output_path),
        )

        return {
            "game_date":       game_date_str,
            "raw_pitches":     len(df_raw),
            "plate_appearances": len(df_pa),
            "output_path":     str(output_path),
        }

    # ===================================================================
    # TASK C — Reentrenamiento del Challenger
    # ===================================================================

    @task(
        task_id="retrain_challenger",
        retries=1,
        retry_delay=timedelta(minutes=5),
        execution_timeout=timedelta(hours=2),
    )
    def retrain_challenger(**context) -> dict:
        """Entrena el modelo Challenger LightGBM y lo registra con alias 'staging'.

        GameDayRetrainingPipeline.run() encapsula:
            1. Actualización incremental del feature store (rolling + platoon)
            2. Entrenamiento AtBatPredictor sobre la ventana histórica completa
            3. Evaluación holdout: Log-Loss ≤ 0.84 (gate hard) y ECE ≤ 0.045 (soft)
            4. Registro en MLflow con alias 'staging' si pasa el gate

        Precondición: ingest_statcast_yesterday completó — la partición
        data/gold/plate_appearances/game_date={game_date}/ existe.
        """
        game_date_str = _get_game_date(context)
        target_date = date.fromisoformat(game_date_str)

        from src.mlops.game_day_retraining import GameDayRetrainingPipeline

        slog.info("retrain_start", game_date=game_date_str)

        pipeline = GameDayRetrainingPipeline.from_env()

        try:
            result = pipeline.run(game_date=target_date)
        except Exception as exc:
            slog.error("retrain_pipeline_exception", game_date=game_date_str, error=str(exc))
            raise AirflowException(
                f"GameDayRetrainingPipeline.run() raised for {game_date_str}: {exc}"
            ) from exc

        if not result.success:
            slog.error(
                "retrain_gate_failed",
                game_date=game_date_str,
                log_loss=result.log_loss,
                alert_message=result.alert_message,
            )
            raise AirflowException(
                f"Retraining quality gate failed for {game_date_str} "
                f"(log_loss={result.log_loss:.5f}): {result.alert_message}"
            )

        slog.info(
            "retrain_complete",
            game_date=game_date_str,
            log_loss=result.log_loss,
            ece_overall=result.ece_overall,
            model_version=result.model_version,
            mlflow_run_id=result.mlflow_run_id,
            challenger_beats_champion=result.challenger_beats_champion,
            training_samples=result.training_samples,
            validation_samples=result.validation_samples,
        )

        return {
            "game_date":               game_date_str,
            "success":                 result.success,
            "log_loss":                result.log_loss,
            "ece_overall":             result.ece_overall,
            "model_version":           result.model_version,
            "mlflow_run_id":           result.mlflow_run_id,
            "challenger_beats_champion": result.challenger_beats_champion,
            "training_samples":        result.training_samples,
            "validation_samples":      result.validation_samples,
            "elapsed_seconds":         result.elapsed_seconds,
        }

    # ===================================================================
    # TASK D — AutoPromoter Gate
    # ===================================================================

    @task(
        task_id="auto_promote",
        trigger_rule=TriggerRule.ALL_SUCCESS,  # solo si retrain_challenger pasó el gate
        retries=1,
        retry_delay=timedelta(minutes=5),
        execution_timeout=timedelta(minutes=15),
    )
    def auto_promote(**context) -> dict:
        """Evalúa el challenger vs. champion y despliega en ECS Fargate si procede.

        Gate 1: Δlog-loss = champion_log_loss - challenger_log_loss ≥ 0.01
        Gate 2: challenger_ece_overall ≤ 0.035

        Si ambos gates pasan:
            - Reasigna alias 'production' en MLflow Registry al challenger.
            - Etiqueta el champion anterior como 'retired'.
            - ECS Fargate: register_task_definition + update_service(forceNewDeployment=True).

        Si algún gate falla → PromotionResult(promoted=False) — producción sin cambios.
        PromotionError (fallo de MLflow o ECS API) → tarea fallida, requiere revisión manual.
        """
        from src.mlops.auto_promoter import AutoPromoter, PromotionError

        slog.info("auto_promote_start")

        promoter = AutoPromoter()

        try:
            result = promoter.promote_if_better()
        except PromotionError as exc:
            slog.error("auto_promote_error", error=str(exc))
            raise AirflowException(
                f"AutoPromoter raised PromotionError (MLflow/ECS API failure): {exc}"
            ) from exc

        if result.promoted:
            slog.info(
                "auto_promote_deployed",
                challenger_version=result.challenger_version,
                previous_champion=result.champion_version,
                delta_log_loss=result.delta_log_loss,
                ecs_task_arn=result.ecs_task_definition_arn,
            )
        else:
            slog.info(
                "auto_promote_skipped",
                reason=result.reason,
                challenger_version=result.challenger_version,
                delta_log_loss=result.delta_log_loss,
            )

        return {
            "promoted":           result.promoted,
            "reason":             result.reason,
            "champion_version":   result.champion_version,
            "challenger_version": result.challenger_version,
            "delta_log_loss":     result.delta_log_loss,
            "ecs_task_arn":       result.ecs_task_definition_arn,
        }

    # ===================================================================
    # Wiring
    # ===================================================================
    #
    # Grafo de dependencias:
    #
    #   close_experiment_observations  (independiente, corre en paralelo)
    #
    #   ingest_statcast_yesterday
    #       └──→ retrain_challenger
    #                └──→ auto_promote
    #
    # Ambas ramas son terminales — el DAG tiene éxito cuando ambas completan.
    # ===================================================================

    close_task = close_experiment_observations()
    ingest_task = ingest_statcast_yesterday()
    retrain_task = retrain_challenger()
    promote_task = auto_promote()

    # Cadena secuencial de entrenamiento
    ingest_task >> retrain_task >> promote_task

    # close_task es independiente — no hay dependencia explícita con la cadena
    # de entrenamiento. Airflow marcará el DAG como SUCCESS cuando AMBAS ramas
    # (close_task y promote_task) hayan completado con éxito.
    # Si se necesita un fan-in explícito, añadir un task de notificación final:
    # [close_task, promote_task] >> notify_completion_task


# ---------------------------------------------------------------------------
# Instancia del DAG (requerido por Airflow para el scheduler)
# ---------------------------------------------------------------------------

dag_instance = mlb_nightly_training()
