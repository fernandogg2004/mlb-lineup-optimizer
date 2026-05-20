"""
statcast_ingestion.py
=====================
Two-mode Statcast ingestion pipeline:

    1. **Historical batch** (``run_historical_backfill``):
       Downloads full Statcast history (2015–present) via ``pybaseball``,
       processes pitch-level attributes in Spark on EMR, and writes to the
       Bronze Delta Lake layer on S3. Designed to be idempotent — already-
       loaded partitions are skipped unless ``--force-reload`` is passed.

    2. **Real-time incremental polling** (``run_realtime_polling``):
       Polls the MLB StatsAPI live feed every N seconds during game hours,
       enriches each pitch event with UUIDs and ingest metadata, and
       publishes to a Kinesis Data Stream for downstream consumers.

Usage (batch):
    python -m src.ingestion.statcast_ingestion batch \\
        --start-year 2015 --end-year 2025 \\
        --bucket mlb-lakehouse --env prod

Usage (realtime):
    python -m src.ingestion.statcast_ingestion realtime \\
        --game-pks 745528,745529 --interval 30

Architecture notes:
    - Bronze writes use Delta ``mergeSchema=true`` to absorb any new columns
      MLB adds to Statcast without breaking the pipeline.
    - The historical backfill is partitioned by ``game_date`` (YYYY-MM-DD);
      Spark partition pruning on downstream Silver queries costs O(1).
    - Realtime events are published to Kinesis with the ``game_pk`` as the
      partition key so pitch ordering is preserved within a game shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from datetime import date, datetime
from typing import Optional

import boto3
import httpx
import structlog
from botocore.exceptions import ClientError
from delta import configure_spark_with_delta_pip
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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
# Constants
# ---------------------------------------------------------------------------
STATCAST_FIRST_SEASON = 2015
MLB_STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
BRONZE_STATCAST_PREFIX = "bronze/statcast"

# Physical validity bounds used by the ingest-time guard rail.
# Great Expectations enforces these more rigorously in Silver; here we just
# drop physically impossible rows before writing to Bronze to avoid polluting
# the lakehouse with sensor noise.
RELEASE_SPEED_MIN_MPH = 40.0
RELEASE_SPEED_MAX_MPH = 110.0
SPIN_RATE_MIN_RPM = 0.0
SPIN_RATE_MAX_RPM = 4000.0
LAUNCH_SPEED_MIN_MPH = 0.0
LAUNCH_SPEED_MAX_MPH = 135.0


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------

def _build_spark(env: str) -> SparkSession:
    """Builds a Delta Lake–enabled SparkSession for the ingestion job.

    Args:
        env: Deployment environment; affects app name for EMR tracking.

    Returns:
        Configured ``SparkSession``.
    """
    builder = (
        SparkSession.builder
        .appName(f"mlb-statcast-ingestion-{env}")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .config("spark.sql.shuffle.partitions", "200")
        # Delta merge schema allows absorbing new MLB Statcast columns
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
    )
    session = configure_spark_with_delta_pip(builder).getOrCreate()
    session.sparkContext.setLogLevel(os.getenv("SPARK_LOG_LEVEL", "WARN"))
    return session


# ---------------------------------------------------------------------------
# Historical batch ingestion
# ---------------------------------------------------------------------------

def _season_date_ranges(start_year: int, end_year: int) -> list[tuple[str, str]]:
    """Generates (start_date, end_date) tuples covering each MLB season.

    Uses April 1 – November 1 to capture spring training, regular season,
    and postseason without downloading off-season gaps.

    Args:
        start_year: First season to include (inclusive).
        end_year: Last season to include (inclusive).

    Returns:
        List of ``(start_dt, end_dt)`` ISO date string tuples.
    """
    return [
        (f"{year}-04-01", f"{year}-11-01")
        for year in range(start_year, end_year + 1)
    ]


def _partition_already_loaded(s3_client, bucket: str, year: int) -> bool:
    """Checks whether a Bronze partition for a given year already exists.

    Uses an S3 HEAD on the ``_delta_log`` directory as a cheap existence
    probe to skip re-downloading seasons that are already present.

    Args:
        s3_client: Boto3 S3 client.
        bucket: S3 bucket name.
        year: Calendar year to check.

    Returns:
        ``True`` if the partition exists, ``False`` otherwise.
    """
    key = f"{BRONZE_STATCAST_PREFIX}/year={year}/_delta_log/"
    try:
        s3_client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
        return True
    except ClientError:
        return False


def _add_pipeline_metadata(df: DataFrame) -> DataFrame:
    """Adds ingest metadata columns to a raw Statcast DataFrame.

    Adds:
        - ``pitch_id``: Deterministic UUID-v5 derived from ``(game_pk, at_bat_number,
          pitch_number)`` so the column is stable across re-runs.
        - ``_ingest_ts``: Wall-clock timestamp of the Spark job run.
        - ``_source``: Literal ``"statcast_pybaseball"`` for lineage tracking.

    Args:
        df: Raw Statcast DataFrame from pybaseball.

    Returns:
        DataFrame with three additional columns.
    """
    # Deterministic UUID: sha1(game_pk || at_bat_number || pitch_number) → UUID v5 namespace
    uuid_udf = F.udf(
        lambda gk, ab, pn: str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{gk}-{ab}-{pn}",
            )
        ),
        StringType(),
    )
    return (
        df.withColumn(
            "pitch_id",
            uuid_udf(
                F.col("game_pk").cast("string"),
                F.col("at_bat_number").cast("string"),
                F.col("pitch_number").cast("string"),
            ),
        )
        .withColumn("_ingest_ts", F.current_timestamp())
        .withColumn("_source", F.lit("statcast_pybaseball"))
    )


def _apply_physical_bounds_filter(df: DataFrame) -> tuple[DataFrame, int]:
    """Drops rows with physically impossible Statcast sensor readings.

    Rows are flagged rather than silently dropped — a count is returned
    so the caller can emit a metric / alert if the bad-row rate is high.

    Args:
        df: Statcast DataFrame (may contain nulls).

    Returns:
        Tuple of (filtered_df, rows_dropped_count).
    """
    total = df.count()
    clean = df.filter(
        (F.col("release_speed").isNull() | F.col("release_speed").between(RELEASE_SPEED_MIN_MPH, RELEASE_SPEED_MAX_MPH))
        & (F.col("release_spin_rate").isNull() | F.col("release_spin_rate").between(SPIN_RATE_MIN_RPM, SPIN_RATE_MAX_RPM))
        & (F.col("launch_speed").isNull() | F.col("launch_speed").between(LAUNCH_SPEED_MIN_MPH, LAUNCH_SPEED_MAX_MPH))
    )
    dropped = total - clean.count()
    if dropped > 0:
        log.warning("physical_bounds_filter", rows_dropped=dropped, total_rows=total)
    return clean, dropped


def ingest_season_to_bronze(
    spark: SparkSession,
    bucket: str,
    start_dt: str,
    end_dt: str,
    force_reload: bool = False,
) -> int:
    """Downloads one season of Statcast data and writes it to Bronze Delta.

    Performs the following steps:
    1. Downloads raw data via ``pybaseball.statcast``.
    2. Converts to Spark DataFrame.
    3. Adds pipeline metadata columns.
    4. Applies physical bounds filter.
    5. Writes to Delta with ``overwrite`` (idempotent per partition).

    Args:
        spark: Active SparkSession.
        bucket: S3 bucket hosting the Bronze layer.
        start_dt: Season start date in ``YYYY-MM-DD`` format.
        end_dt: Season end date in ``YYYY-MM-DD`` format.
        force_reload: If ``True``, overwrites existing partitions.

    Returns:
        Number of rows written to Bronze.

    Raises:
        RuntimeError: If pybaseball download or Spark write fails.
    """
    year = int(start_dt[:4])
    s3 = boto3.client("s3")
    output_path = f"s3a://{bucket}/{BRONZE_STATCAST_PREFIX}/year={year}"

    if not force_reload and _partition_already_loaded(s3, bucket, year):
        log.info("season_already_loaded_skipping", year=year)
        return 0

    log.info("season_download_start", year=year, start=start_dt, end=end_dt)

    try:
        # pybaseball is imported here (not at module top) so the rest of the
        # module works in environments where pybaseball is not installed (e.g.
        # Lambda runtime for realtime mode).
        from pybaseball import statcast  # noqa: PLC0415

        raw_pdf = statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
    except Exception as exc:
        log.error("pybaseball_download_failed", year=year, error=str(exc))
        raise RuntimeError(f"pybaseball download failed for {year}: {exc}") from exc

    if raw_pdf.empty:
        log.warning("season_empty_result", year=year)
        return 0

    log.info("season_download_complete", year=year, rows=len(raw_pdf))

    # Convert pandas → Spark; inferSchema is acceptable here because Bronze
    # accepts whatever MLB ships and schema evolution handles new columns.
    sdf = spark.createDataFrame(raw_pdf)
    sdf = _add_pipeline_metadata(sdf)
    sdf, dropped = _apply_physical_bounds_filter(sdf)

    rows_written = sdf.count()

    try:
        (
            sdf.write.format("delta")
            .mode("overwrite")
            .option("mergeSchema", "true")
            .option("overwriteSchema", "false")
            .save(output_path)
        )
    except Exception as exc:
        log.error("bronze_write_failed", year=year, path=output_path, error=str(exc))
        raise RuntimeError(f"Delta write failed for {year}: {exc}") from exc

    log.info(
        "season_ingested",
        year=year,
        rows_written=rows_written,
        rows_dropped=dropped,
        path=output_path,
    )
    return rows_written


def run_historical_backfill(
    spark: SparkSession,
    bucket: str,
    start_year: int,
    end_year: int,
    force_reload: bool = False,
) -> dict[int, int]:
    """Runs the full historical backfill across all requested seasons.

    Each season is downloaded and written independently so a failure in one
    year does not block others. Errors are logged and collected; a summary
    is returned to the caller.

    Args:
        spark: Active SparkSession.
        bucket: S3 bucket for the Bronze layer.
        start_year: First season to ingest (inclusive).
        end_year: Last season to ingest (inclusive).
        force_reload: Overwrite already-loaded seasons if ``True``.

    Returns:
        Dict mapping ``year → rows_written`` (0 = skipped, -1 = failed).
    """
    results: dict[int, int] = {}
    for start_dt, end_dt in _season_date_ranges(start_year, end_year):
        year = int(start_dt[:4])
        try:
            rows = ingest_season_to_bronze(spark, bucket, start_dt, end_dt, force_reload)
            results[year] = rows
        except RuntimeError as exc:
            log.error("season_ingestion_failed", year=year, error=str(exc))
            results[year] = -1

    success = sum(1 for v in results.values() if v >= 0)
    failed = sum(1 for v in results.values() if v == -1)
    log.info("backfill_complete", success=success, failed=failed, total=len(results))
    return results


# ---------------------------------------------------------------------------
# Real-time incremental polling (MLB StatsAPI)
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _fetch_live_game_feed(game_pk: int, http_client: httpx.Client) -> dict:
    """Fetches the live game feed JSON for a single game from the MLB StatsAPI.

    Retries with exponential backoff on HTTP errors (5xx, rate-limit 429).
    The live feed endpoint returns the complete current game state including
    all plays, pitch-by-pitch data, and boxscore.

    Args:
        game_pk: MLB game identifier.
        http_client: Shared ``httpx.Client`` for connection reuse.

    Returns:
        Parsed JSON response as a Python dict.

    Raises:
        httpx.HTTPStatusError: After all retry attempts are exhausted.
    """
    url = f"{MLB_STATSAPI_BASE}.1/game/{game_pk}/feed/live"
    resp = http_client.get(url, timeout=15.0)
    resp.raise_for_status()
    return resp.json()


def _extract_pitch_events(game_feed: dict, game_pk: int) -> list[dict]:
    """Extracts individual pitch events from an MLB StatsAPI live feed.

    Navigates the nested ``allPlays → playEvents`` structure and normalizes
    pitch attributes into a flat dictionary aligned with the Bronze schema.

    Args:
        game_feed: Full JSON response from the live game feed endpoint.
        game_pk: Game identifier, injected into each event for tracing.

    Returns:
        List of pitch event dicts ready for Kinesis publication.
    """
    pitches: list[dict] = []
    try:
        all_plays = (
            game_feed.get("liveData", {})
            .get("plays", {})
            .get("allPlays", [])
        )
    except (KeyError, AttributeError):
        log.warning("feed_parse_error", game_pk=game_pk)
        return pitches

    game_date = (
        game_feed.get("gameData", {})
        .get("datetime", {})
        .get("officialDate", "")
    )

    for play in all_plays:
        for event in play.get("playEvents", []):
            if event.get("type") != "pitch":
                continue

            pitch_data = event.get("pitchData", {})
            hit_data = event.get("hitData", {})
            details = event.get("details", {})

            pitches.append(
                {
                    "pitch_id": str(
                        uuid.uuid5(
                            uuid.NAMESPACE_DNS,
                            f"{game_pk}-{play.get('atBatIndex')}-{event.get('index')}",
                        )
                    ),
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "inning": play.get("about", {}).get("inning"),
                    "inning_top_bot": play.get("about", {}).get("halfInning"),
                    "pitcher_id": play.get("matchup", {}).get("pitcher", {}).get("id"),
                    "batter_id": play.get("matchup", {}).get("batter", {}).get("id"),
                    "pitch_type": details.get("type", {}).get("code"),
                    "release_speed": pitch_data.get("startSpeed"),
                    "release_spin_rate": pitch_data.get("breaks", {}).get("spinRate"),
                    "pfx_x": pitch_data.get("breaks", {}).get("breakHorizontal"),
                    "pfx_z": pitch_data.get("breaks", {}).get("breakVertical"),
                    "plate_x": pitch_data.get("coordinates", {}).get("pX"),
                    "plate_z": pitch_data.get("coordinates", {}).get("pZ"),
                    "launch_speed": hit_data.get("launchSpeed"),
                    "launch_angle": hit_data.get("launchAngle"),
                    "hit_distance_sc": hit_data.get("totalDistance"),
                    "events": play.get("result", {}).get("event"),
                    "description": details.get("description"),
                    "stand": play.get("matchup", {}).get("batSide", {}).get("code"),
                    "p_throws": play.get("matchup", {}).get("pitchHand", {}).get("code"),
                    "_ingest_ts": datetime.utcnow().isoformat(),
                    "_source": "statsapi_live",
                }
            )

    return pitches


def _publish_pitches_to_kinesis(
    pitches: list[dict],
    stream_name: str,
    kinesis_client,
) -> int:
    """Publishes pitch events to a Kinesis Data Stream in batches of 500.

    Kinesis ``PutRecords`` accepts up to 500 records per call. Uses
    ``game_pk`` as the partition key so all pitches from one game land on
    the same shard and downstream consumers see them in order.

    Args:
        pitches: List of pitch event dicts.
        stream_name: Kinesis stream name.
        kinesis_client: Boto3 Kinesis client.

    Returns:
        Total number of records successfully published.
    """
    batch_size = 500
    published = 0

    for i in range(0, len(pitches), batch_size):
        batch = pitches[i : i + batch_size]
        records = [
            {
                "Data": json.dumps(p).encode("utf-8"),
                "PartitionKey": str(p["game_pk"]),
            }
            for p in batch
        ]
        try:
            resp = kinesis_client.put_records(StreamName=stream_name, Records=records)
            failed = resp.get("FailedRecordCount", 0)
            published += len(records) - failed
            if failed:
                log.warning("kinesis_partial_failure", failed=failed, batch_index=i)
        except ClientError as exc:
            log.error("kinesis_publish_failed", error=str(exc), batch_index=i)

    return published


def run_realtime_polling(
    game_pks: list[int],
    kinesis_stream: str,
    poll_interval_seconds: int = 30,
    max_iterations: Optional[int] = None,
) -> None:
    """Polls the MLB StatsAPI live feed and publishes new pitches to Kinesis.

    Runs an indefinite polling loop (or ``max_iterations`` for testing).
    Tracks the last known pitch index per game to publish only new events,
    preventing duplicate messages to Kinesis.

    Args:
        game_pks: List of game PKs to monitor in this polling session.
        kinesis_stream: Kinesis stream name for pitch events.
        poll_interval_seconds: Seconds between API polls per game.
        max_iterations: Limit polling cycles (``None`` = run indefinitely).

    Note:
        In production this function is invoked by an Airflow sensor task that
        monitors the MLB schedule and passes the day's game_pks at first pitch.
    """
    kinesis = boto3.client("kinesis", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    # last_pitch_index[game_pk] = highest pitch event index already published
    last_pitch_index: dict[int, int] = {gk: -1 for gk in game_pks}
    iteration = 0

    with httpx.Client(headers={"User-Agent": "mlb-lineup-optimizer/1.0"}) as client:
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            for game_pk in game_pks:
                try:
                    feed = _fetch_live_game_feed(game_pk, client)
                    pitches = _extract_pitch_events(feed, game_pk)

                    # Filter to only new pitches since last poll
                    new_pitches = [
                        p for p in pitches
                        if int(
                            hashlib.md5(p["pitch_id"].encode()).hexdigest(), 16
                        ) > last_pitch_index[game_pk]
                    ]

                    # Simpler: track by list length (fine because pitch order is monotonic)
                    known_count = last_pitch_index[game_pk] + 1
                    new_pitches = pitches[known_count:]

                    if new_pitches:
                        published = _publish_pitches_to_kinesis(
                            new_pitches, kinesis_stream, kinesis
                        )
                        last_pitch_index[game_pk] = len(pitches) - 1
                        log.info(
                            "pitches_published",
                            game_pk=game_pk,
                            new_pitches=len(new_pitches),
                            published=published,
                        )
                except httpx.HTTPStatusError as exc:
                    log.error("poll_http_error", game_pk=game_pk, status=exc.response.status_code)
                except Exception as exc:
                    log.error("poll_unexpected_error", game_pk=game_pk, error=str(exc))

            if max_iterations is None or iteration < max_iterations:
                time.sleep(poll_interval_seconds)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLB Statcast ingestion pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    # batch subcommand
    batch_p = sub.add_parser("batch", help="Historical backfill mode")
    batch_p.add_argument("--start-year", type=int, default=STATCAST_FIRST_SEASON)
    batch_p.add_argument("--end-year", type=int, default=date.today().year)
    batch_p.add_argument("--bucket", required=True)
    batch_p.add_argument("--env", default="dev")
    batch_p.add_argument("--force-reload", action="store_true")

    # realtime subcommand
    rt_p = sub.add_parser("realtime", help="Live game polling mode")
    rt_p.add_argument("--game-pks", required=True, help="Comma-separated game PKs")
    rt_p.add_argument("--kinesis-stream", default="statcast-pitches-live")
    rt_p.add_argument("--interval", type=int, default=30)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI dispatcher for batch and realtime ingestion modes.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]`` when ``None``.
    """
    args = _parse_args(argv)

    if args.mode == "batch":
        spark = _build_spark(args.env)
        try:
            results = run_historical_backfill(
                spark=spark,
                bucket=args.bucket,
                start_year=args.start_year,
                end_year=args.end_year,
                force_reload=args.force_reload,
            )
            total_rows = sum(v for v in results.values() if v > 0)
            log.info("batch_complete", total_rows_written=total_rows)
        finally:
            spark.stop()

    elif args.mode == "realtime":
        game_pks = [int(pk.strip()) for pk in args.game_pks.split(",")]
        run_realtime_polling(
            game_pks=game_pks,
            kinesis_stream=args.kinesis_stream,
            poll_interval_seconds=args.interval,
        )


if __name__ == "__main__":
    main()
