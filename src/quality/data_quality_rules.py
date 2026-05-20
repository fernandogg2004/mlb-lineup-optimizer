"""
data_quality_rules.py
=====================
Declarative Great Expectations suite for the Silver ``plate_appearances`` table.

Design philosophy:
    - Expectations encode domain knowledge as executable documentation.
      Any data scientist joining the project reads this file to understand
      what constitutes valid baseball physics data.
    - Rules are divided into three tiers:
        * CRITICAL — violations route the entire batch to quarantine; the
          pipeline halts and pages on-call. These guard physical invariants
          that can never be wrong (e.g. launch_speed cannot exceed 135 mph).
        * WARNING — violations increment a metric counter and are flagged in
          the ``_ge_validated`` column as ``"WARN"``, but the batch proceeds.
          These handle edge cases like rare weather extremes.
        * INFORMATIONAL — tracked for drift monitoring but never block the
          pipeline. These surface gradual distribution shifts to the ML team.
    - The suite is registered against the Delta table path so Airflow can run
      it as a post-ingest checkpoint with a single call to ``run_checkpoint``.

Usage:
    # Build and save the suite to the GE Data Context
    python -m src.quality.data_quality_rules build --context-root ./ge_context

    # Validate a specific game_date partition
    python -m src.quality.data_quality_rules validate \\
        --context-root ./ge_context --game-date 2024-07-15

    # Run as a library (called from Airflow DAG)
    from src.quality.data_quality_rules import run_validation_checkpoint
    result = run_validation_checkpoint(context_root="./ge_context", game_date="2024-07-15")

Physical validity references:
    - launch_speed: MLB-measured range 0–122 mph (highest ever: 122.4 mph,
      Giancarlo Stanton). We allow 135 mph upper bound for sensor noise margin.
    - spin_rate: Typical range 1500–3500 rpm. Knuckleball ~1000 rpm is the
      practical minimum; 4000 rpm is the theoretical ceiling for any pitch type.
    - launch_angle: -90° (straight down) to +90° (straight up) by definition.
      Playable range is roughly -45° to +60°.
    - temperature_f: MLB games are played in 35–105 °F range. Outside this,
      the game is postponed; a weather snapshot outside this range is suspect.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from great_expectations.core import ExpectationSuite
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import FileDataContext
from great_expectations.exceptions import DataContextError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Suite and datasource identifiers
# ---------------------------------------------------------------------------
SUITE_NAME = "silver_plate_appearances_suite"
DATASOURCE_NAME = "silver_plate_appearances_datasource"
DATA_ASSET_NAME = "silver_plate_appearances"
CHECKPOINT_NAME = "silver_pa_checkpoint"


# ---------------------------------------------------------------------------
# Physical bounds constants (also used in statcast_ingestion guard rail)
# ---------------------------------------------------------------------------
class PhysicalBounds:
    """Physical validity bounds for Statcast measurements.

    These are hard limits derived from the known physics of baseball and
    the documented measurement ranges of the Trackman / Hawk-Eye systems.
    Anything outside these ranges is a sensor error, not a real event.
    """

    # Pitch release speed (mph)
    RELEASE_SPEED_MIN: float = 40.0   # slowest legal eephus pitch ~40 mph
    RELEASE_SPEED_MAX: float = 110.0  # fastest ever: Aroldis Chapman 105.8 mph

    # Pitch spin rate (rpm)
    SPIN_RATE_MIN: float = 800.0      # knuckleball practical floor
    SPIN_RATE_MAX: float = 3800.0     # theoretical max for any pitch type

    # Exit velocity / launch speed (mph)
    LAUNCH_SPEED_MIN: float = 0.0
    LAUNCH_SPEED_MAX: float = 130.0   # highest ever: ~122 mph + margin

    # Launch angle (degrees)
    LAUNCH_ANGLE_MIN: float = -75.0
    LAUNCH_ANGLE_MAX: float = 75.0

    # Spray angle (degrees from center field line; negative = left, positive = right)
    SPRAY_ANGLE_MIN: float = -50.0
    SPRAY_ANGLE_MAX: float = 50.0

    # Hit distance (feet)
    HIT_DISTANCE_MIN: float = 0.0
    HIT_DISTANCE_MAX: float = 600.0   # Babe Ruth's longest ~575 ft; 600 ft is impossible

    # Weather
    TEMPERATURE_F_MIN: float = 20.0   # below this, game would be postponed
    TEMPERATURE_F_MAX: float = 115.0
    HUMIDITY_MIN: float = 0.0
    HUMIDITY_MAX: float = 100.0
    WIND_SPEED_MIN: float = 0.0
    WIND_SPEED_MAX: float = 60.0      # above ~50 mph games are delayed
    PRESSURE_INHG_MIN: float = 26.0   # ~high altitude / extreme low pressure
    PRESSURE_INHG_MAX: float = 32.0   # sea level high pressure

    # Inning
    INNING_MIN: int = 1
    INNING_MAX: int = 26             # longest MLB game: 26 innings (1920)


# ---------------------------------------------------------------------------
# GE Data Context setup
# ---------------------------------------------------------------------------

def build_data_context(context_root: str) -> FileDataContext:
    """Creates or opens a Great Expectations FileDataContext.

    The context uses filesystem backends stored in ``context_root`` so the
    suite definitions, validation results, and data docs are version-controlled
    alongside the pipeline code.

    Args:
        context_root: Absolute or relative path to the GE project root.

    Returns:
        Initialized ``FileDataContext`` instance.

    Raises:
        DataContextError: If the context cannot be created or opened.
    """
    context_path = Path(context_root)
    context_path.mkdir(parents=True, exist_ok=True)

    try:
        context = FileDataContext.create(project_root_dir=str(context_path))
        log.info("ge_context_created", path=str(context_path))
    except DataContextError:
        context = FileDataContext(context_root_dir=str(context_path))
        log.info("ge_context_opened", path=str(context_path))

    return context


def register_spark_datasource(context: FileDataContext, delta_table_path: str) -> None:
    """Registers the Silver Delta table as a Spark datasource in the GE context.

    Uses ``SparkDFDatasource`` which accepts a PySpark DataFrame at validation
    time, so the datasource definition is decoupled from any specific batch.

    Args:
        context: Initialized GE FileDataContext.
        delta_table_path: S3 path to the Delta table root (e.g.
            ``s3a://mlb-lakehouse/silver/plate_appearances``).

    Raises:
        Exception: Re-raised if datasource registration fails.
    """
    datasource_config = {
        "name": DATASOURCE_NAME,
        "class_name": "Datasource",
        "module_name": "great_expectations.datasource",
        "execution_engine": {
            "module_name": "great_expectations.execution_engine",
            "class_name": "SparkDFExecutionEngine",
        },
        "data_connectors": {
            "runtime_data_connector": {
                "class_name": "RuntimeDataConnector",
                "module_name": "great_expectations.datasource.data_connector",
                "batch_identifiers": ["game_date", "run_id"],
            }
        },
    }
    try:
        context.add_datasource(**datasource_config)
        log.info("datasource_registered", name=DATASOURCE_NAME)
    except Exception as exc:
        if "already exists" in str(exc).lower():
            log.info("datasource_already_registered", name=DATASOURCE_NAME)
        else:
            log.error("datasource_registration_failed", error=str(exc))
            raise


# ---------------------------------------------------------------------------
# Expectation Suite builder
# ---------------------------------------------------------------------------

def build_plate_appearances_suite(context: FileDataContext) -> ExpectationSuite:
    """Constructs the full declarative Expectation Suite for Silver plate_appearances.

    Organizes expectations into three comment-annotated tiers (CRITICAL /
    WARNING / INFORMATIONAL) so the pipeline can route failures differently
    based on severity.

    Critical expectations use ``meta={"severity": "CRITICAL"}`` so the
    Airflow post-processing step can filter by severity when deciding whether
    to halt or continue.

    Args:
        context: Initialized GE FileDataContext.

    Returns:
        Saved ``ExpectationSuite`` instance.
    """
    suite = context.add_or_update_expectation_suite(expectation_suite_name=SUITE_NAME)
    # validator = context.get_validator(...) — only needed when running against real data

    # We build expectations programmatically on the suite object directly
    # (not via validator) so this function can run without a live Spark session.
    suite.expectations = []  # reset to avoid duplicates on re-runs

    b = PhysicalBounds()

    # ==========================================================================
    # TIER 1 — CRITICAL: Physical invariants that cannot be violated
    # ==========================================================================

    # --- Schema completeness ---
    _add(suite, "expect_column_to_exist", {"column": "pa_id"},
         "CRITICAL", "Primary key pa_id must always be present")

    _add(suite, "expect_column_values_to_not_be_null",
         {"column": "pa_id", "mostly": 1.0},
         "CRITICAL", "pa_id null = unidentifiable PA, cannot be ingested")

    _add(suite, "expect_column_values_to_not_be_null",
         {"column": "game_pk", "mostly": 1.0},
         "CRITICAL", "game_pk null = PA cannot be linked to a game")

    _add(suite, "expect_column_values_to_not_be_null",
         {"column": "batter_id", "mostly": 1.0},
         "CRITICAL", "batter_id null = feature lookup impossible for this PA")

    _add(suite, "expect_column_values_to_not_be_null",
         {"column": "pitcher_id", "mostly": 1.0},
         "CRITICAL", "pitcher_id null = matchup features impossible")

    _add(suite, "expect_column_values_to_not_be_null",
         {"column": "pa_result", "mostly": 1.0},
         "CRITICAL", "pa_result is the label for supervised learning; null = unusable row")

    # --- PA result categorical validity ---
    _add(suite, "expect_column_values_to_be_in_set",
         {
             "column": "pa_result",
             "value_set": [
                 "single", "double", "triple", "home_run",
                 "walk", "intent_walk", "hit_by_pitch",
                 "strikeout", "strikeout_double_play",
                 "field_out", "grounded_into_double_play",
                 "double_play", "triple_play",
                 "fielders_choice", "fielders_choice_out",
                 "force_out", "sac_fly", "sac_bunt",
                 "sac_fly_double_play", "catcher_interf",
             ],
             "mostly": 0.99,
         },
         "CRITICAL", "Unknown pa_result breaks label encoding downstream")

    # --- Batter / pitcher handedness ---
    _add(suite, "expect_column_values_to_be_in_set",
         {"column": "batter_stand", "value_set": ["L", "R", "S"], "mostly": 1.0},
         "CRITICAL", "Handedness outside L/R/S is a data contract violation")

    _add(suite, "expect_column_values_to_be_in_set",
         {"column": "pitcher_throws", "value_set": ["L", "R"], "mostly": 1.0},
         "CRITICAL", "Pitcher handedness outside L/R is a data contract violation")

    # --- Inning range ---
    _add(suite, "expect_column_values_to_be_between",
         {"column": "inning", "min_value": b.INNING_MIN, "max_value": b.INNING_MAX, "mostly": 1.0},
         "CRITICAL", f"Inning outside {b.INNING_MIN}–{b.INNING_MAX} is physically impossible")

    # --- Launch speed physical bounds ---
    _add(suite, "expect_column_values_to_be_between",
         {
             "column": "launch_speed",
             "min_value": b.LAUNCH_SPEED_MIN,
             "max_value": b.LAUNCH_SPEED_MAX,
             "mostly": 0.999,
         },
         "CRITICAL", f"launch_speed outside {b.LAUNCH_SPEED_MIN}–{b.LAUNCH_SPEED_MAX} mph = sensor error")

    # --- Launch angle physical bounds ---
    _add(suite, "expect_column_values_to_be_between",
         {
             "column": "launch_angle",
             "min_value": b.LAUNCH_ANGLE_MIN,
             "max_value": b.LAUNCH_ANGLE_MAX,
             "mostly": 0.999,
         },
         "CRITICAL", f"launch_angle outside {b.LAUNCH_ANGLE_MIN}–{b.LAUNCH_ANGLE_MAX}° is geometrically impossible")

    # --- Pitch release speed ---
    _add(suite, "expect_column_values_to_be_between",
         {
             "column": "last_pitch_speed",
             "min_value": b.RELEASE_SPEED_MIN,
             "max_value": b.RELEASE_SPEED_MAX,
             "mostly": 0.999,
         },
         "CRITICAL", f"last_pitch_speed outside {b.RELEASE_SPEED_MIN}–{b.RELEASE_SPEED_MAX} mph = Trackman sensor error")

    # --- Spin rate ---
    _add(suite, "expect_column_values_to_be_between",
         {
             "column": "last_pitch_spin_rate",
             "min_value": b.SPIN_RATE_MIN,
             "max_value": b.SPIN_RATE_MAX,
             "mostly": 0.999,
         },
         "CRITICAL", f"spin_rate outside {b.SPIN_RATE_MIN}–{b.SPIN_RATE_MAX} rpm = sensor fault")

    # --- PA uniqueness ---
    _add(suite, "expect_column_values_to_be_unique",
         {"column": "pa_id"},
         "CRITICAL", "Duplicate pa_id = pipeline bug (re-ingest without deduplication)")

    # --- Validated flag values ---
    _add(suite, "expect_column_values_to_be_in_set",
         {"column": "_ge_validated", "value_set": ["PASS", "WARN", "QUARANTINE"], "mostly": 1.0},
         "CRITICAL", "_ge_validated must be set by the Silver transform; missing = pipeline bug")

    # ==========================================================================
    # TIER 2 — WARNING: Expected but not guaranteed; alert if exceeded
    # ==========================================================================

    # --- Null tolerance for Statcast-derived features ---
    # Statcast only tracks batted balls; PAs ending in strikeout/walk have
    # null launch data. Null rate above 60% suggests a join failure, not
    # just natural Ks and BBs.
    _add(suite, "expect_column_values_to_not_be_null",
         {"column": "launch_speed", "mostly": 0.4},  # ~40% of PAs have batted ball data
         "WARNING", "launch_speed null >60% = possible Bronze-Silver join failure")

    _add(suite, "expect_column_values_to_not_be_null",
         {"column": "xwoba", "mostly": 0.35},
         "WARNING", "xwoba null >65% = Statcast expected stats API may be lagging")

    # --- Weather join completeness ---
    _add(suite, "expect_column_values_to_not_be_null",
         {"column": "temperature_f", "mostly": 0.95},
         "WARNING", ">5% weather nulls = weather Lambda may be failing for some venues")

    # --- Temperature range for actual game conditions ---
    _add(suite, "expect_column_values_to_be_between",
         {
             "column": "temperature_f",
             "min_value": b.TEMPERATURE_F_MIN,
             "max_value": b.TEMPERATURE_F_MAX,
             "mostly": 0.999,
         },
         "WARNING", f"temperature outside {b.TEMPERATURE_F_MIN}–{b.TEMPERATURE_F_MAX}°F during a game is anomalous")

    _add(suite, "expect_column_values_to_be_between",
         {
             "column": "humidity_pct",
             "min_value": b.HUMIDITY_MIN,
             "max_value": b.HUMIDITY_MAX,
             "mostly": 0.999,
         },
         "WARNING", "humidity outside 0–100% is physically impossible")

    _add(suite, "expect_column_values_to_be_between",
         {
             "column": "wind_speed_mph",
             "min_value": b.WIND_SPEED_MIN,
             "max_value": b.WIND_SPEED_MAX,
             "mostly": 0.999,
         },
         "WARNING", f"wind speed >{b.WIND_SPEED_MAX} mph is above game-delay threshold")

    _add(suite, "expect_column_values_to_be_between",
         {
             "column": "pressure_inhg",
             "min_value": b.PRESSURE_INHG_MIN,
             "max_value": b.PRESSURE_INHG_MAX,
             "mostly": 0.999,
         },
         "WARNING", f"pressure outside {b.PRESSURE_INHG_MIN}–{b.PRESSURE_INHG_MAX} inHg = sensor or units error")

    # --- Hit distance ---
    _add(suite, "expect_column_values_to_be_between",
         {
             "column": "hit_distance",
             "min_value": b.HIT_DISTANCE_MIN,
             "max_value": b.HIT_DISTANCE_MAX,
             "mostly": 0.999,
         },
         "WARNING", f"hit_distance >{b.HIT_DISTANCE_MAX} ft is physically impossible")

    # --- Spray angle ---
    _add(suite, "expect_column_values_to_be_between",
         {
             "column": "spray_angle",
             "min_value": b.SPRAY_ANGLE_MIN,
             "max_value": b.SPRAY_ANGLE_MAX,
             "mostly": 0.999,
         },
         "WARNING", f"spray_angle outside {b.SPRAY_ANGLE_MIN}–{b.SPRAY_ANGLE_MAX}° is outside fair territory")

    # ==========================================================================
    # TIER 3 — INFORMATIONAL: Distribution drift monitoring
    # ==========================================================================

    # Row count — extremely low counts per partition = ingestion gap
    _add(suite, "expect_table_row_count_to_be_between",
         {"min_value": 100, "max_value": 5000},
         "INFORMATIONAL", "Typical game has 600–900 PAs; <100 = likely partial ingest")

    # Season completeness: expect both hands represented
    _add(suite, "expect_column_proportion_of_unique_values_to_be_between",
         {"column": "pitcher_throws", "min_proportion": 0.05, "max_proportion": 0.95},
         "INFORMATIONAL", "Game-day files dominated by one pitcher hand is unusual")

    # xwOBA mean should stay near league average (~0.310–0.340)
    _add(suite, "expect_column_mean_to_be_between",
         {"column": "xwoba", "min_value": 0.240, "max_value": 0.420},
         "INFORMATIONAL", "xwOBA mean drift outside 0.240–0.420 may indicate model or data shift")

    # Launch speed mean historically ~87–92 mph
    _add(suite, "expect_column_mean_to_be_between",
         {"column": "launch_speed", "min_value": 75.0, "max_value": 100.0},
         "INFORMATIONAL", "Mean launch_speed drift = possible Statcast calibration change")

    context.save_expectation_suite(suite, discard_failed_expectations=False)
    log.info("suite_saved", suite_name=SUITE_NAME, expectation_count=len(suite.expectations))
    return suite


def _add(
    suite: ExpectationSuite,
    expectation_type: str,
    kwargs: dict[str, Any],
    severity: str,
    notes: str,
) -> None:
    """Helper to add an expectation to a suite with standard metadata.

    Args:
        suite: GE ExpectationSuite being constructed.
        expectation_type: GE expectation type string.
        kwargs: Expectation-specific keyword arguments.
        severity: One of ``"CRITICAL"``, ``"WARNING"``, ``"INFORMATIONAL"``.
        notes: Human-readable explanation of *why* this rule exists.
    """
    from great_expectations.core import ExpectationConfiguration  # noqa: PLC0415

    config = ExpectationConfiguration(
        expectation_type=expectation_type,
        kwargs=kwargs,
        meta={
            "severity": severity,
            "notes": notes,
            "added_by": "data_quality_rules.py",
            "added_at": datetime.utcnow().isoformat(),
        },
    )
    suite.add_expectation(config, send_usage_event=False)


# ---------------------------------------------------------------------------
# Checkpoint setup and validation runner
# ---------------------------------------------------------------------------

def register_checkpoint(context: FileDataContext) -> None:
    """Registers a SimpleCheckpoint for the Silver PA suite.

    The checkpoint is what Airflow calls after each Silver transformation.
    It binds the suite to the datasource and configures action list:
    1. store_validation_result — persists to GE validations store.
    2. update_data_docs — refreshes the HTML data docs site.
    3. slack_alert (optional) — posts to #data-quality Slack on CRITICAL failure.

    Args:
        context: Initialized GE FileDataContext.
    """
    checkpoint_config = {
        "name": CHECKPOINT_NAME,
        "config_version": 1.0,
        "class_name": "SimpleCheckpoint",
        "run_name_template": "%Y%m%d-%H%M%S-silver-pa-validation",
        "validations": [
            {
                "batch_request": {
                    "datasource_name": DATASOURCE_NAME,
                    "data_connector_name": "runtime_data_connector",
                    "data_asset_name": DATA_ASSET_NAME,
                    "batch_identifiers": {
                        "game_date": "RUNTIME_PLACEHOLDER",
                        "run_id": "RUNTIME_PLACEHOLDER",
                    },
                },
                "expectation_suite_name": SUITE_NAME,
            }
        ],
        "action_list": [
            {
                "name": "store_validation_result",
                "action": {"class_name": "StoreValidationResultAction"},
            },
            {
                "name": "update_data_docs",
                "action": {"class_name": "UpdateDataDocsAction", "site_names": []},
            },
        ],
    }
    try:
        context.add_or_update_checkpoint(**checkpoint_config)
        log.info("checkpoint_registered", name=CHECKPOINT_NAME)
    except Exception as exc:
        log.error("checkpoint_registration_failed", error=str(exc))
        raise


def run_validation_checkpoint(
    context_root: str,
    game_date: str,
    spark_dataframe,  # pyspark.sql.DataFrame
) -> dict[str, Any]:
    """Validates a Silver plate_appearances partition against the suite.

    Intended to be called from an Airflow task immediately after the Bronze →
    Silver dbt/Spark transformation completes for a given game_date.

    Args:
        context_root: Path to the GE FileDataContext directory.
        game_date: Partition date being validated (``YYYY-MM-DD``).
        spark_dataframe: PySpark DataFrame of the Silver partition to validate.

    Returns:
        Dict with keys:
            - ``"success"``: ``True`` if all CRITICAL expectations passed.
            - ``"critical_failures"``: List of failed CRITICAL expectation dicts.
            - ``"warning_failures"``: List of failed WARNING expectation dicts.
            - ``"statistics"``: GE validation statistics dict.

    Raises:
        RuntimeError: If the GE checkpoint run fails unexpectedly.
    """
    context = build_data_context(context_root)
    run_id = f"{game_date}-{datetime.utcnow().strftime('%H%M%S')}"

    batch_request = RuntimeBatchRequest(
        datasource_name=DATASOURCE_NAME,
        data_connector_name="runtime_data_connector",
        data_asset_name=DATA_ASSET_NAME,
        runtime_parameters={"batch_data": spark_dataframe},
        batch_identifiers={"game_date": game_date, "run_id": run_id},
    )

    try:
        result = context.run_checkpoint(
            checkpoint_name=CHECKPOINT_NAME,
            validations=[
                {
                    "batch_request": batch_request,
                    "expectation_suite_name": SUITE_NAME,
                }
            ],
        )
    except Exception as exc:
        log.error("checkpoint_run_failed", game_date=game_date, error=str(exc))
        raise RuntimeError(f"GE checkpoint failed for {game_date}: {exc}") from exc

    # Parse results into structured output the Airflow DAG can act on
    validation_result = list(result.run_results.values())[0]["validation_result"]
    stats = validation_result["statistics"]
    results_list = validation_result["results"]

    critical_failures = [
        r for r in results_list
        if not r["success"] and r.get("expectation_config", {}).get("meta", {}).get("severity") == "CRITICAL"
    ]
    warning_failures = [
        r for r in results_list
        if not r["success"] and r.get("expectation_config", {}).get("meta", {}).get("severity") == "WARNING"
    ]

    overall_success = len(critical_failures) == 0

    log.info(
        "validation_complete",
        game_date=game_date,
        success=overall_success,
        critical_failures=len(critical_failures),
        warning_failures=len(warning_failures),
        evaluated=stats.get("evaluated_expectations"),
        successful=stats.get("successful_expectations"),
    )

    if critical_failures:
        log.critical(
            "critical_expectations_failed",
            game_date=game_date,
            failures=[
                {
                    "expectation": r["expectation_config"]["expectation_type"],
                    "column": r["expectation_config"]["kwargs"].get("column"),
                    "notes": r["expectation_config"].get("meta", {}).get("notes"),
                }
                for r in critical_failures
            ],
        )

    return {
        "success": overall_success,
        "critical_failures": critical_failures,
        "warning_failures": warning_failures,
        "statistics": stats,
        "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Great Expectations suite manager for Silver PA table")
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build and save the expectation suite")
    build_p.add_argument("--context-root", default="./ge_context", dest="context_root")
    build_p.add_argument("--delta-path", default=os.getenv("SILVER_PA_PATH", ""),
                         dest="delta_path", help="S3 path to the Silver Delta table")

    val_p = sub.add_parser("validate", help="Run checkpoint validation for a game date")
    val_p.add_argument("--context-root", default="./ge_context", dest="context_root")
    val_p.add_argument("--game-date", required=True, dest="game_date")
    val_p.add_argument("--delta-path", required=True, dest="delta_path")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for suite management operations.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]`` when ``None``.
    """
    args = _parse_args(argv)
    context = build_data_context(args.context_root)

    if args.command == "build":
        register_spark_datasource(context, args.delta_path)
        suite = build_plate_appearances_suite(context)
        register_checkpoint(context)
        log.info("suite_build_complete",
                 suite=SUITE_NAME,
                 expectation_count=len(suite.expectations))

    elif args.command == "validate":
        from pyspark.sql import SparkSession  # noqa: PLC0415
        from delta import configure_spark_with_delta_pip  # noqa: PLC0415

        spark = configure_spark_with_delta_pip(
            SparkSession.builder
            .appName("ge-validation")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        ).getOrCreate()

        df = (
            spark.read.format("delta")
            .load(args.delta_path)
            .filter(f"game_date = '{args.game_date}'")
        )
        result = run_validation_checkpoint(
            context_root=args.context_root,
            game_date=args.game_date,
            spark_dataframe=df,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "critical_failures"}, indent=2))
        spark.stop()


if __name__ == "__main__":
    main()
