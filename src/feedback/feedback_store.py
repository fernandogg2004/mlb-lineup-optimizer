"""
feedback_store.py  —  Capas 2 y 3 del sistema de retroalimentación cualitativa.
================================================================================
Persiste los flags del bench coach en Parquet particionado por game_date y
expone dos análisis batch:

Capa 2 — Re-ponderación de etiquetas (ejecutar semanalmente o por el pipeline
         de reentrenamiento diario):
    get_sample_weights(from_date, to_date) → DataFrame (player_id, game_date,
                                              sample_weight)
    El resultado se pasa como `sample_weight` a LightGBM en el reentrenamiento.
    Un jugador con flag HARD physical_illness(-0.7) ese día → sample_weight=0.30,
    sus PAs de ese día cuentan menos en el ajuste del modelo.

Capa 3 — Propuestas de features (ejecutar semanalmente):
    analyze_recurring_flags(lookback_days=30, threshold=3) → list[FeatureProposal]
    Si el mismo tipo de flag aparece ≥3 veces para el mismo jugador en 30 días,
    genera una propuesta de feature binaria para revisión humana.
    Ejemplo: "recurring_physical_illness_30d" para el jugador 592450.

Schema del store (normalizado, una fila por flag por submission):
    submission_id   str     UUID-16 del envío
    game_id         str     ID del partido
    game_date       str     YYYY-MM-DD (clave de partición)
    submitted_by    str     "bench_coach" | "manager" | etc.
    submission_type str     "pre_game" | "post_game"
    player_id       i64
    player_name     str
    flag_type       str     FlagType.value
    severity        str     "soft" | "hard"
    weight_override f64     [-1.0, 1.0]
    notes           str     texto libre del coach (máx 280 chars)
    ai_utility_score       i64 nullable
    lineup_decision_quality i64 nullable
    hidden_context  str     texto libre (máx 280 chars)
    model_was_right bool nullable
    why_model_was_wrong str texto libre
    submitted_at    str     ISO 8601 UTC

Ruta de almacenamiento:
    {store_path}/game_date={YYYY-MM-DD}/{submission_id}.parquet
    Configurable vía variable de entorno FEEDBACK_STORE_PATH.
    Por defecto: data/gold/qualitative_feedback/
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import polars as pl
import structlog

from app.schemas import FeatureProposal, PlayerFlagRequest, QualitativeFeedbackRequest

logger = structlog.get_logger(__name__)

# Escala de severidad replicada aquí para no importar override_engine (evita circular)
_SEVERITY_SCALE: dict[str, float] = {"soft": 0.5, "hard": 1.0}

# Peso de muestra mínimo para no excluir nunca completamente un ejemplo
_SAMPLE_WEIGHT_MIN: float = 0.05
_SAMPLE_WEIGHT_MAX: float = 2.0

# Umbral para propuestas de features (Layer 3)
_RECURRING_FLAG_THRESHOLD: int = 3


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------


@dataclass
class FeedbackStoreConfig:
    """Configuración del store de feedback.

    Attributes:
        store_path: Ruta raíz del store Parquet.
            Puede ser local o una ruta S3 (s3://mlb-lakehouse/gold/qualitative_feedback).
        partition_col: Columna de partición (solo game_date soportado actualmente).
    """

    store_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("FEEDBACK_STORE_PATH", "data/gold/qualitative_feedback")
        )
    )

    @classmethod
    def from_env(cls) -> "FeedbackStoreConfig":
        return cls(
            store_path=Path(
                os.environ.get("FEEDBACK_STORE_PATH", "data/gold/qualitative_feedback")
            )
        )


# ---------------------------------------------------------------------------
# Store principal
# ---------------------------------------------------------------------------


class FeedbackStore:
    """Persiste feedback cualitativo y expone análisis para re-ponderación y features.

    Args:
        config: Configuración del store.
    """

    def __init__(self, config: FeedbackStoreConfig | None = None) -> None:
        self._config = config or FeedbackStoreConfig.from_env()
        self._config.store_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "feedback_store_initialized",
            path=str(self._config.store_path),
        )

    @classmethod
    def from_env(cls) -> "FeedbackStore":
        return cls(FeedbackStoreConfig.from_env())

    # ------------------------------------------------------------------
    # Escritura
    # ------------------------------------------------------------------

    def save(
        self,
        feedback: QualitativeFeedbackRequest,
        submission_id: str,
    ) -> int:
        """Persiste el feedback en Parquet particionado por game_date.

        Normaliza el payload: una fila por flag. Los campos de evaluación
        post-partido (ai_utility_score, etc.) se repiten en cada fila del
        mismo submission para simplificar las queries analíticas.

        Args:
            feedback: Payload validado del request de feedback.
            submission_id: UUID-16 generado en el endpoint antes de llamar a save.

        Returns:
            Número de filas escritas (= número de flags en el payload).
        """
        if not feedback.player_flags:
            logger.warning("feedback_save_empty_flags", submission_id=submission_id)
            return 0

        submitted_at = datetime.now(tz=timezone.utc).isoformat()
        rows = [
            {
                "submission_id": submission_id,
                "game_id": feedback.game_id,
                "game_date": feedback.game_date,
                "submitted_by": feedback.submitted_by,
                "submission_type": feedback.submission_type.value,
                "player_id": flag.player_id,
                "player_name": flag.player_name,
                "flag_type": flag.flag_type.value,
                "severity": flag.severity.value,
                "weight_override": flag.weight_override,
                "notes": flag.notes or "",
                "ai_utility_score": feedback.ai_utility_score,
                "lineup_decision_quality": feedback.lineup_decision_quality,
                "hidden_context": feedback.hidden_context or "",
                "model_was_right": feedback.model_was_right,
                "why_model_was_wrong": feedback.why_model_was_wrong or "",
                "submitted_at": submitted_at,
            }
            for flag in feedback.player_flags
        ]

        df = pl.DataFrame(rows, schema=_FEEDBACK_SCHEMA)

        partition_dir = self._config.store_path / f"game_date={feedback.game_date}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        output_path = partition_dir / f"{submission_id}.parquet"
        df.write_parquet(output_path)

        logger.info(
            "feedback_saved",
            submission_id=submission_id,
            game_id=feedback.game_id,
            game_date=feedback.game_date,
            n_flags=len(rows),
            path=str(output_path),
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Lectura — Layer 1 (overrides en tiempo real)
    # ------------------------------------------------------------------

    def get_active_flags(
        self,
        game_date: str,
        player_ids: list[int] | None = None,
    ) -> list[PlayerFlagRequest]:
        """Carga los flags pre-partido activos para el game_date dado.

        Solo devuelve flags con submission_type="pre_game" ya que son los
        que deben influir en la optimización antes del primer pitch.

        Args:
            game_date: Fecha del partido en formato YYYY-MM-DD.
            player_ids: Si se especifica, filtra a estos player_ids.

        Returns:
            Lista de PlayerFlagRequest listos para pasarse a apply_qualitative_overrides.
        """
        df = self._read_partition(game_date)
        if df is None or df.is_empty():
            return []

        df = df.filter(pl.col("submission_type") == "pre_game")
        if player_ids:
            df = df.filter(pl.col("player_id").is_in(player_ids))

        flags: list[PlayerFlagRequest] = []
        for row in df.iter_rows(named=True):
            try:
                flags.append(PlayerFlagRequest(
                    player_id=row["player_id"],
                    player_name=row["player_name"],
                    flag_type=row["flag_type"],
                    severity=row["severity"],
                    weight_override=row["weight_override"],
                    notes=row["notes"] or None,
                ))
            except Exception as exc:
                logger.warning(
                    "flag_deserialization_error",
                    row=row,
                    error=str(exc),
                )
        logger.info(
            "active_flags_loaded",
            game_date=game_date,
            n_flags=len(flags),
        )
        return flags

    # ------------------------------------------------------------------
    # Lectura — Layer 2 (sample weights para reentrenamiento)
    # ------------------------------------------------------------------

    def get_sample_weights(
        self,
        from_date: str,
        to_date: str,
    ) -> pl.DataFrame:
        """Calcula sample weights por (player_id, game_date) para reentrenamiento.

        Para cada jugador × día con flags, el sample_weight es:
            weight = max(MIN, min(MAX, 1.0 + sum(weight × severity_scale)))
        Un flag HARD physical_illness(-0.7) → weight = max(0.05, 1.0 - 0.7) = 0.30.
        Sus PAs de ese día cuentan 30% en el ajuste del modelo.

        Uso en reentrenamiento:
            weights_df = store.get_sample_weights(from_date, to_date)
            pa_df = pa_df.join(weights_df, on=["player_id", "game_date"], how="left")
            pa_df = pa_df.with_columns(
                pl.col("sample_weight").fill_null(1.0)
            )
            X, y, weights = pa_df.to_numpy(...)

        Args:
            from_date: Fecha inicio inclusive (YYYY-MM-DD).
            to_date: Fecha fin inclusive (YYYY-MM-DD).

        Returns:
            DataFrame Polars con columnas: player_id (i64), game_date (str),
            sample_weight (f64). Solo incluye días con flags activos.
        """
        all_dfs: list[pl.DataFrame] = []
        current = date.fromisoformat(from_date)
        end = date.fromisoformat(to_date)

        while current <= end:
            df = self._read_partition(current.isoformat())
            if df is not None and not df.is_empty():
                all_dfs.append(df.select(["player_id", "game_date", "severity", "weight_override"]))
            current += timedelta(days=1)

        if not all_dfs:
            return pl.DataFrame(schema={"player_id": pl.Int64, "game_date": pl.Utf8, "sample_weight": pl.Float64})

        combined = pl.concat(all_dfs)
        weights = (
            combined
            .with_columns(
                (
                    pl.col("weight_override") *
                    pl.col("severity").map_elements(
                        lambda s: _SEVERITY_SCALE.get(s, 1.0),
                        return_dtype=pl.Float64,
                    )
                ).alias("effective_contribution")
            )
            .group_by(["player_id", "game_date"])
            .agg(pl.col("effective_contribution").sum().alias("raw_weight"))
            .with_columns(
                pl.col("raw_weight")
                .clip(_SAMPLE_WEIGHT_MIN - 1.0, _SAMPLE_WEIGHT_MAX - 1.0)
                .add(1.0)
                .clip(_SAMPLE_WEIGHT_MIN, _SAMPLE_WEIGHT_MAX)
                .alias("sample_weight")
            )
            .select(["player_id", "game_date", "sample_weight"])
        )

        logger.info(
            "sample_weights_computed",
            from_date=from_date,
            to_date=to_date,
            n_player_days=len(weights),
        )
        return weights

    # ------------------------------------------------------------------
    # Lectura — Layer 3 (propuestas de features)
    # ------------------------------------------------------------------

    def analyze_recurring_flags(
        self,
        lookback_days: int = 30,
        threshold: int = _RECURRING_FLAG_THRESHOLD,
    ) -> list[FeatureProposal]:
        """Detecta flags recurrentes y genera propuestas de features para revisión.

        Una propuesta se genera cuando el mismo tipo de flag aparece ≥ threshold
        veces para el mismo jugador en los últimos lookback_days días. Las
        propuestas requieren aprobación humana — no se integran automáticamente.

        Args:
            lookback_days: Ventana de análisis en días.
            threshold: Mínimo de ocurrencias para generar propuesta.

        Returns:
            Lista de FeatureProposal para revisión del equipo ML.
        """
        to_date = date.today()
        from_date = to_date - timedelta(days=lookback_days)

        all_dfs: list[pl.DataFrame] = []
        current = from_date
        while current <= to_date:
            df = self._read_partition(current.isoformat())
            if df is not None and not df.is_empty():
                all_dfs.append(
                    df.select(["player_id", "player_name", "flag_type", "game_date"])
                )
            current += timedelta(days=1)

        if not all_dfs:
            return []

        combined = pl.concat(all_dfs)
        recurring = (
            combined
            .group_by(["player_id", "player_name", "flag_type"])
            .agg([
                pl.len().alias("occurrences"),
                pl.col("game_date").min().alias("first_observed"),
                pl.col("game_date").max().alias("last_observed"),
            ])
            .filter(pl.col("occurrences") >= threshold)
            .sort("occurrences", descending=True)
        )

        proposals: list[FeatureProposal] = []
        for row in recurring.iter_rows(named=True):
            feature_name = (
                f"recurring_{row['flag_type']}_last_{lookback_days}d"
            )
            rationale = (
                f"Flag '{row['flag_type']}' observado {row['occurrences']} veces "
                f"para {row['player_name']} entre {row['first_observed']} y "
                f"{row['last_observed']}. Posible señal estructural que merece "
                "una feature binaria en el modelo."
            )
            proposals.append(FeatureProposal(
                player_id=row["player_id"],
                player_name=row["player_name"],
                flag_type=row["flag_type"],
                occurrences=row["occurrences"],
                lookback_days=lookback_days,
                first_observed=row["first_observed"],
                last_observed=row["last_observed"],
                proposed_feature_name=feature_name,
                rationale_es=rationale,
                requires_human_review=True,
            ))

        logger.info(
            "recurring_flags_analyzed",
            lookback_days=lookback_days,
            threshold=threshold,
            proposals_generated=len(proposals),
        )
        return proposals

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
# Schema Polars para validación en escritura
# ---------------------------------------------------------------------------

_FEEDBACK_SCHEMA = {
    "submission_id": pl.Utf8,
    "game_id": pl.Utf8,
    "game_date": pl.Utf8,
    "submitted_by": pl.Utf8,
    "submission_type": pl.Utf8,
    "player_id": pl.Int64,
    "player_name": pl.Utf8,
    "flag_type": pl.Utf8,
    "severity": pl.Utf8,
    "weight_override": pl.Float64,
    "notes": pl.Utf8,
    "ai_utility_score": pl.Int64,
    "lineup_decision_quality": pl.Int64,
    "hidden_context": pl.Utf8,
    "model_was_right": pl.Boolean,
    "why_model_was_wrong": pl.Utf8,
    "submitted_at": pl.Utf8,
}
