"""
experiment_store.py — A/B Experiment Tracker para el Protocolo Manager vs. AI.
================================================================================
Persiste las divergencias entre la recomendación del optimizador y la decisión
real del mánager. Tras registrar el resultado post-partido, calcula estadísticas
acumuladas que permiten medir empíricamente si el sistema agrega valor.

Ciclo de vida de una observación:
    1. Pre-partido: mánager recibe /v1/optimize/lineup → elige una alineación diferente
       → POST /v1/experiment/record-divergence → status="open"
    2. Post-partido: se conoce el resultado real
       → POST /v1/experiment/record-outcome → status="closed"
    3. Análisis: GET /v1/experiment/summary agrega todas las observaciones cerradas

Schema de almacenamiento (una fila por observación):
    observation_id   str     UUID-16 del registro
    game_id          str     ID del partido
    game_date        str     YYYY-MM-DD (clave de partición)
    ai_lineup_ids    str     JSON: lista de 9 player_ids en orden AI
    manager_lineup_ids str   JSON: lista de 9 player_ids en orden del mánager
    ai_expected_runs f64     E[R] estimado por el optimizador para el lineup AI
    actual_runs_scored i64   Carreras reales del lineup del mánager (null hasta cierre)
    innings_played   i64     Entradas jugadas (null hasta cierre)
    divergence_type  str     "order_only" | "player_swap" | "full_override"
    slots_changed    i64     Posiciones del batting order que difieren
    players_changed  i64     Jugadores que el mánager cambió respecto al AI
    manager_rationale str    Texto libre del mánager (máx 280 chars)
    status           str     "open" | "closed"
    submitted_at     str     ISO 8601 UTC
    closed_at        str     ISO 8601 UTC (vacío hasta cierre)

Ruta de almacenamiento:
    {store_path}/game_date={YYYY-MM-DD}/{observation_id}.parquet
    Configurable vía variable de entorno EXPERIMENT_STORE_PATH.
    Por defecto: data/gold/ab_experiment/
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

_MIN_SAMPLE_SIGNIFICANT = 15   # Mínimo de partidos cerrados para estadísticas concluyentes
_OUTCOME_SEARCH_WINDOW = 14    # Días hacia atrás al buscar observaciones por game_id


# ---------------------------------------------------------------------------
# Schema Polars
# ---------------------------------------------------------------------------

_OBSERVATION_SCHEMA: dict[str, type] = {
    "observation_id": pl.Utf8,
    "game_id": pl.Utf8,
    "game_date": pl.Utf8,
    "ai_lineup_ids": pl.Utf8,
    "manager_lineup_ids": pl.Utf8,
    "ai_expected_runs": pl.Float64,
    "actual_runs_scored": pl.Int64,
    "innings_played": pl.Int64,
    "divergence_type": pl.Utf8,
    "slots_changed": pl.Int64,
    "players_changed": pl.Int64,
    "manager_rationale": pl.Utf8,
    "status": pl.Utf8,
    "submitted_at": pl.Utf8,
    "closed_at": pl.Utf8,
}


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------


@dataclass
class ExperimentStoreConfig:
    store_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("EXPERIMENT_STORE_PATH", "data/gold/ab_experiment")
        )
    )

    @classmethod
    def from_env(cls) -> "ExperimentStoreConfig":
        return cls(
            store_path=Path(
                os.environ.get("EXPERIMENT_STORE_PATH", "data/gold/ab_experiment")
            )
        )


# ---------------------------------------------------------------------------
# Helpers de clasificación
# ---------------------------------------------------------------------------


def classify_divergence(
    ai_ids: list[int],
    manager_ids: list[int],
) -> tuple[str, int, int]:
    """Clasifica la divergencia entre el lineup AI y el del mánager.

    Args:
        ai_ids: Lista de 9 player_ids en el orden recomendado por el AI.
        manager_ids: Lista de 9 player_ids en el orden real del mánager.

    Returns:
        (divergence_type, slots_changed, players_changed)
        divergence_type: "order_only" | "player_swap" | "full_override"
        slots_changed: número de posiciones del batting order que difieren
        players_changed: número de jugadores del AI no incluidos por el mánager
    """
    players_changed = len(set(ai_ids) - set(manager_ids))
    slots_changed = sum(1 for a, m in zip(ai_ids, manager_ids) if a != m)

    if players_changed == 0:
        dtype = "order_only"
    elif players_changed == 1:
        dtype = "player_swap"
    else:
        dtype = "full_override"

    return dtype, slots_changed, players_changed


# ---------------------------------------------------------------------------
# Store principal
# ---------------------------------------------------------------------------


class ExperimentStore:
    """Persiste observaciones A/B y expone estadísticas del experimento.

    Args:
        config: Configuración del store.
    """

    def __init__(self, config: ExperimentStoreConfig | None = None) -> None:
        self._config = config or ExperimentStoreConfig.from_env()
        self._config.store_path.mkdir(parents=True, exist_ok=True)
        logger.info("experiment_store_initialized", path=str(self._config.store_path))

    @classmethod
    def from_env(cls) -> "ExperimentStore":
        return cls(ExperimentStoreConfig.from_env())

    # ------------------------------------------------------------------
    # Escritura — registro de divergencia
    # ------------------------------------------------------------------

    def record_divergence(
        self,
        game_id: str,
        game_date: str,
        ai_lineup_ids: list[int],
        manager_lineup_ids: list[int],
        ai_expected_runs: float,
        manager_rationale: str,
        observation_id: str,
    ) -> tuple[str, int, int]:
        """Persiste una observación de divergencia con status='open'.

        Args:
            game_id: ID del partido.
            game_date: Fecha YYYY-MM-DD para la partición.
            ai_lineup_ids: Los 9 player_ids en orden AI (de lineup_player_ids).
            manager_lineup_ids: Los 9 player_ids en orden real del mánager.
            ai_expected_runs: E[R] del optimizador para su lineup.
            manager_rationale: Texto libre del mánager (puede ser vacío).
            observation_id: UUID-16 generado en el endpoint.

        Returns:
            (divergence_type, slots_changed, players_changed)
        """
        dtype, slots_changed, players_changed = classify_divergence(
            ai_lineup_ids, manager_lineup_ids
        )
        submitted_at = datetime.now(tz=timezone.utc).isoformat()

        row = {
            "observation_id": observation_id,
            "game_id": game_id,
            "game_date": game_date,
            "ai_lineup_ids": json.dumps(ai_lineup_ids),
            "manager_lineup_ids": json.dumps(manager_lineup_ids),
            "ai_expected_runs": ai_expected_runs,
            "actual_runs_scored": None,
            "innings_played": None,
            "divergence_type": dtype,
            "slots_changed": slots_changed,
            "players_changed": players_changed,
            "manager_rationale": manager_rationale or "",
            "status": "open",
            "submitted_at": submitted_at,
            "closed_at": "",
        }

        df = pl.DataFrame([row], schema=_OBSERVATION_SCHEMA)
        partition_dir = self._config.store_path / f"game_date={game_date}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        output_path = partition_dir / f"{observation_id}.parquet"
        df.write_parquet(output_path)

        logger.info(
            "divergence_recorded",
            observation_id=observation_id,
            game_id=game_id,
            game_date=game_date,
            divergence_type=dtype,
            slots_changed=slots_changed,
            players_changed=players_changed,
        )
        return dtype, slots_changed, players_changed

    # ------------------------------------------------------------------
    # Escritura — registro de resultado post-partido
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        game_id: str,
        actual_runs_scored: int,
        innings_played: int = 9,
    ) -> tuple[int, float]:
        """Cierra las observaciones abiertas de un game_id con el resultado real.

        Escanea las últimas OUTCOME_SEARCH_WINDOW particiones para encontrar
        observaciones con status='open' y game_id coincidente.

        Args:
            game_id: ID del partido.
            actual_runs_scored: Carreras reales del lineup del mánager.
            innings_played: Entradas jugadas (9 en partidos normales).

        Returns:
            (observations_updated, ai_expected_runs)
            ai_expected_runs es el del primer match encontrado (o 0.0 si ninguno).
        """
        today = date.today()
        closed_at = datetime.now(tz=timezone.utc).isoformat()
        total_updated = 0
        first_ai_er = 0.0

        for days_back in range(_OUTCOME_SEARCH_WINDOW + 1):
            check_date = (today - timedelta(days=days_back)).isoformat()
            partition_dir = self._config.store_path / f"game_date={check_date}"
            if not partition_dir.exists():
                continue

            for parquet_file in partition_dir.glob("*.parquet"):
                df = pl.read_parquet(parquet_file, hive_partitioning=False)
                open_matches = df.filter(
                    (pl.col("game_id") == game_id) & (pl.col("status") == "open")
                )
                if open_matches.is_empty():
                    continue

                n_matched = len(open_matches)
                if first_ai_er == 0.0:
                    first_ai_er = float(open_matches["ai_expected_runs"][0])

                updated_df = df.with_columns([
                    pl.when(
                        (pl.col("game_id") == game_id) & (pl.col("status") == "open")
                    )
                    .then(pl.lit(actual_runs_scored))
                    .otherwise(pl.col("actual_runs_scored"))
                    .cast(pl.Int64)
                    .alias("actual_runs_scored"),

                    pl.when(
                        (pl.col("game_id") == game_id) & (pl.col("status") == "open")
                    )
                    .then(pl.lit(innings_played))
                    .otherwise(pl.col("innings_played"))
                    .cast(pl.Int64)
                    .alias("innings_played"),

                    pl.when(
                        (pl.col("game_id") == game_id) & (pl.col("status") == "open")
                    )
                    .then(pl.lit("closed"))
                    .otherwise(pl.col("status"))
                    .alias("status"),

                    pl.when(
                        (pl.col("game_id") == game_id) & (pl.col("status") == "open")
                    )
                    .then(pl.lit(closed_at))
                    .otherwise(pl.col("closed_at"))
                    .alias("closed_at"),
                ])

                updated_df.write_parquet(parquet_file)
                total_updated += n_matched
                logger.info(
                    "outcome_recorded_partial",
                    game_id=game_id,
                    game_date=check_date,
                    n_updated=n_matched,
                    actual_runs_scored=actual_runs_scored,
                )

        logger.info(
            "outcome_recorded",
            game_id=game_id,
            actual_runs_scored=actual_runs_scored,
            innings_played=innings_played,
            total_updated=total_updated,
        )
        return total_updated, first_ai_er

    # ------------------------------------------------------------------
    # Lectura — estadísticas del experimento A/B
    # ------------------------------------------------------------------

    def get_summary(self, lookback_days: int = 90) -> dict:
        """Calcula estadísticas A/B sobre observaciones cerradas en el período.

        Args:
            lookback_days: Ventana de análisis en días hacia atrás desde hoy.

        Returns:
            Dict con claves: total_closed_games, by_divergence_type,
            avg_ai_expected_runs, avg_actual_runs_scored, manager_advantage_runs,
            sample_too_small, interpretation_es.
        """
        today = date.today()
        from_date = today - timedelta(days=lookback_days)

        all_dfs: list[pl.DataFrame] = []
        current = from_date
        while current <= today:
            df = self._read_partition(current.isoformat())
            if df is not None and not df.is_empty():
                closed = df.filter(pl.col("status") == "closed")
                if not closed.is_empty():
                    all_dfs.append(
                        closed.select([
                            "game_id",
                            "ai_expected_runs",
                            "actual_runs_scored",
                            "divergence_type",
                            "slots_changed",
                            "players_changed",
                        ])
                    )
            current += timedelta(days=1)

        if not all_dfs:
            return {
                "total_closed_games": 0,
                "by_divergence_type": {"order_only": 0, "player_swap": 0, "full_override": 0},
                "avg_ai_expected_runs": None,
                "avg_actual_runs_scored": None,
                "manager_advantage_runs": None,
                "sample_too_small": True,
                "interpretation_es": (
                    f"Sin datos en los últimos {lookback_days} días. "
                    "Registra divergencias con POST /v1/experiment/record-divergence "
                    "y resultados con POST /v1/experiment/record-outcome."
                ),
            }

        combined = pl.concat(all_dfs)
        total = len(combined)

        by_type = {
            "order_only": int(
                combined.filter(pl.col("divergence_type") == "order_only").height
            ),
            "player_swap": int(
                combined.filter(pl.col("divergence_type") == "player_swap").height
            ),
            "full_override": int(
                combined.filter(pl.col("divergence_type") == "full_override").height
            ),
        }

        avg_ai = combined["ai_expected_runs"].mean()
        avg_actual_series = combined["actual_runs_scored"].drop_nulls()
        avg_actual = avg_actual_series.mean() if len(avg_actual_series) > 0 else None

        manager_advantage: Optional[float] = None
        if avg_actual is not None and avg_ai is not None:
            manager_advantage = round(float(avg_actual) - float(avg_ai), 3)

        sample_too_small = total < _MIN_SAMPLE_SIGNIFICANT
        interpretation_es = _build_interpretation_es(
            total=total,
            avg_ai=float(avg_ai) if avg_ai is not None else None,
            avg_actual=float(avg_actual) if avg_actual is not None else None,
            manager_advantage=manager_advantage,
            sample_too_small=sample_too_small,
            lookback_days=lookback_days,
        )

        logger.info(
            "experiment_summary_computed",
            lookback_days=lookback_days,
            total=total,
            sample_too_small=sample_too_small,
        )

        return {
            "total_closed_games": total,
            "by_divergence_type": by_type,
            "avg_ai_expected_runs": round(float(avg_ai), 3) if avg_ai is not None else None,
            "avg_actual_runs_scored": round(float(avg_actual), 3) if avg_actual is not None else None,
            "manager_advantage_runs": manager_advantage,
            "sample_too_small": sample_too_small,
            "interpretation_es": interpretation_es,
        }

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _read_partition(self, game_date: str) -> Optional[pl.DataFrame]:
        """Lee todos los archivos Parquet de la partición game_date."""
        partition_dir = self._config.store_path / f"game_date={game_date}"
        if not partition_dir.exists():
            return None
        parquet_files = list(partition_dir.glob("*.parquet"))
        if not parquet_files:
            return None
        dfs = [pl.read_parquet(f, hive_partitioning=False) for f in parquet_files]
        return pl.concat(dfs)


# ---------------------------------------------------------------------------
# Helpers de interpretación
# ---------------------------------------------------------------------------


def _build_interpretation_es(
    total: int,
    avg_ai: Optional[float],
    avg_actual: Optional[float],
    manager_advantage: Optional[float],
    sample_too_small: bool,
    lookback_days: int,
) -> str:
    if sample_too_small:
        needed = max(0, _MIN_SAMPLE_SIGNIFICANT - total)
        return (
            f"Muestra insuficiente: {total} partido(s) registrado(s) en los últimos "
            f"{lookback_days} días. Se necesitan {needed} más para estadísticas concluyentes."
        )

    if manager_advantage is None or avg_ai is None or avg_actual is None:
        return "Datos incompletos — registra resultados con POST /v1/experiment/record-outcome."

    abs_adv = abs(manager_advantage)
    season_proj = round(abs_adv * 162, 1)

    if abs_adv < 0.05:
        verdict = "El mánager y el AI producen resultados estadísticamente equivalentes."
    elif manager_advantage > 0:
        verdict = (
            f"El mánager supera la expectativa del AI en +{abs_adv:.2f} C/partido "
            f"cuando diverge ({season_proj} carreras en 162 partidos)."
        )
    else:
        verdict = (
            f"El lineup del AI habría producido {abs_adv:.2f} C/partido más "
            f"en promedio ({season_proj} carreras en 162 partidos)."
        )

    return (
        f"{verdict} Basado en {total} partidos con resultado registrado "
        f"(últimos {lookback_days} días). "
        f"Media AI: {avg_ai:.2f} E[C] | Media real: {avg_actual:.2f} C."
    )
