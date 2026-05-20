"""
catalog_setup.py
================
Initializes the MLB Lineup Optimizer Lakehouse: creates the Delta Lake
table schemas in the Bronze / Silver / Gold layers and registers them in
AWS Glue Data Catalog so any Spark, Athena, or dbt workload can discover
them without hard-coded paths.

Execution:
    python -m src.catalog.catalog_setup --env prod --bucket mlb-lakehouse

Design decisions:
    - Delta Lake is the single storage format across all layers to get
      ACID semantics, schema evolution, and time-travel for free.
    - Partition columns are chosen to match the most common query predicates
      (game_date, venue_id) so Spark prunes partitions on every downstream
      read, keeping scan costs near O(1) per game-day query.
    - Glue registration uses `LOCATION` pointing to S3 Delta paths; Glue
      stores only metadata. Delta itself is the source of truth.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

import boto3
import structlog
from botocore.exceptions import ClientError
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Logging setup
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
# Data contracts — canonical schemas per layer
# ---------------------------------------------------------------------------

# Bronze: raw Statcast pitch-level data, unmodified from source.
BRONZE_STATCAST_SCHEMA = StructType(
    [
        StructField("pitch_id", StringType(), nullable=False),       # UUID generated at ingest
        StructField("game_pk", LongType(), nullable=False),
        StructField("game_date", StringType(), nullable=False),       # ISO date string, partition key
        StructField("inning", ShortType(), nullable=True),
        StructField("inning_top_bot", StringType(), nullable=True),
        StructField("pitcher_id", IntegerType(), nullable=True),
        StructField("batter_id", IntegerType(), nullable=True),
        StructField("pitch_type", StringType(), nullable=True),
        StructField("release_speed", FloatType(), nullable=True),     # mph
        StructField("release_spin_rate", FloatType(), nullable=True), # rpm
        StructField("pfx_x", FloatType(), nullable=True),            # horizontal break, inches
        StructField("pfx_z", FloatType(), nullable=True),            # induced vertical break, inches
        StructField("release_pos_x", FloatType(), nullable=True),
        StructField("release_pos_y", FloatType(), nullable=True),
        StructField("release_pos_z", FloatType(), nullable=True),
        StructField("plate_x", FloatType(), nullable=True),
        StructField("plate_z", FloatType(), nullable=True),
        StructField("launch_speed", FloatType(), nullable=True),      # exit velocity, mph
        StructField("launch_angle", FloatType(), nullable=True),      # degrees
        StructField("spray_angle", FloatType(), nullable=True),       # degrees, field position
        StructField("hit_distance_sc", FloatType(), nullable=True),   # projected distance, feet
        StructField("events", StringType(), nullable=True),           # PA outcome
        StructField("description", StringType(), nullable=True),      # pitch outcome
        StructField("zone", ShortType(), nullable=True),
        StructField("stand", StringType(), nullable=True),            # batter handedness L/R
        StructField("p_throws", StringType(), nullable=True),         # pitcher handedness L/R
        StructField("home_team", StringType(), nullable=True),
        StructField("away_team", StringType(), nullable=True),
        StructField("venue_id", StringType(), nullable=True),
        StructField("_ingest_ts", TimestampType(), nullable=False),   # pipeline metadata
        StructField("_source", StringType(), nullable=False),         # "statcast" | "statsapi"
    ]
)

# Silver: cleaned, validated, joined plate-appearance grain (one row = one PA).
SILVER_PLATE_APPEARANCES_SCHEMA = StructType(
    [
        StructField("pa_id", StringType(), nullable=False),
        StructField("game_pk", LongType(), nullable=False),
        StructField("game_date", StringType(), nullable=False),
        StructField("venue_id", StringType(), nullable=False),
        StructField("inning", ShortType(), nullable=False),
        StructField("inning_top_bot", StringType(), nullable=False),
        StructField("batter_id", IntegerType(), nullable=False),
        StructField("batter_name", StringType(), nullable=True),
        StructField("batter_stand", StringType(), nullable=False),
        StructField("pitcher_id", IntegerType(), nullable=False),
        StructField("pitcher_name", StringType(), nullable=True),
        StructField("pitcher_throws", StringType(), nullable=False),
        StructField("pitch_count_in_pa", ShortType(), nullable=True),
        # Aggregate Statcast physics for the last pitch of the PA
        StructField("last_pitch_type", StringType(), nullable=True),
        StructField("last_pitch_speed", FloatType(), nullable=True),
        StructField("last_pitch_spin_rate", FloatType(), nullable=True),
        StructField("last_pfx_x", FloatType(), nullable=True),
        StructField("last_pfx_z", FloatType(), nullable=True),
        # PA outcome
        StructField("pa_result", StringType(), nullable=False),       # hit/walk/strikeout/...
        StructField("hit_type", StringType(), nullable=True),         # single/double/triple/hr
        StructField("launch_speed", FloatType(), nullable=True),
        StructField("launch_angle", FloatType(), nullable=True),
        StructField("spray_angle", FloatType(), nullable=True),
        StructField("hit_distance", FloatType(), nullable=True),
        StructField("xba", FloatType(), nullable=True),               # expected batting average
        StructField("xslg", FloatType(), nullable=True),              # expected slugging
        StructField("xwoba", FloatType(), nullable=True),             # expected wOBA
        # Weather snapshot joined at PA time
        StructField("temperature_f", FloatType(), nullable=True),
        StructField("humidity_pct", FloatType(), nullable=True),
        StructField("wind_speed_mph", FloatType(), nullable=True),
        StructField("wind_direction_deg", FloatType(), nullable=True),
        StructField("pressure_inhg", FloatType(), nullable=True),
        StructField("weather_condition", StringType(), nullable=True),
        # Metadata
        StructField("season", ShortType(), nullable=False),
        StructField("_silver_ts", TimestampType(), nullable=False),
        StructField("_ge_validated", StringType(), nullable=False),   # "PASS" | "QUARANTINE"
    ]
)

# Gold: pre-aggregated rolling batter features (7 / 15 / 30 / 60-day windows).
GOLD_BATTER_FEATURES_SCHEMA = StructType(
    [
        StructField("feature_date", StringType(), nullable=False),    # partition key
        StructField("batter_id", IntegerType(), nullable=False),
        StructField("batter_name", StringType(), nullable=True),
        StructField("season", ShortType(), nullable=False),
        StructField("pa_7d", IntegerType(), nullable=True),
        StructField("pa_15d", IntegerType(), nullable=True),
        StructField("pa_30d", IntegerType(), nullable=True),
        StructField("pa_60d", IntegerType(), nullable=True),
        StructField("avg_launch_speed_7d", FloatType(), nullable=True),
        StructField("avg_launch_speed_30d", FloatType(), nullable=True),
        StructField("avg_launch_angle_7d", FloatType(), nullable=True),
        StructField("avg_launch_angle_30d", FloatType(), nullable=True),
        StructField("xwoba_7d", FloatType(), nullable=True),
        StructField("xwoba_15d", FloatType(), nullable=True),
        StructField("xwoba_30d", FloatType(), nullable=True),
        StructField("xwoba_60d", FloatType(), nullable=True),
        StructField("k_rate_30d", FloatType(), nullable=True),
        StructField("bb_rate_30d", FloatType(), nullable=True),
        StructField("hr_rate_30d", FloatType(), nullable=True),
        StructField("babip_30d", FloatType(), nullable=True),
        # Platoon splits (vs. RHP / LHP)
        StructField("xwoba_vs_rhp_30d", FloatType(), nullable=True),
        StructField("xwoba_vs_lhp_30d", FloatType(), nullable=True),
        StructField("_gold_ts", TimestampType(), nullable=False),
    ]
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class LakehouseConfig:
    """Runtime configuration for the catalog setup job.

    Args:
        bucket: S3 bucket name that hosts the lakehouse layers.
        glue_database: AWS Glue database where tables are registered.
        region: AWS region for Glue and S3.
        env: Deployment environment tag (dev / staging / prod).
        spark_master: Spark master URL; defaults to local for testing.
    """

    bucket: str
    glue_database: str = "mlb_lakehouse"
    region: str = "us-east-1"
    env: str = "dev"
    spark_master: str = field(default_factory=lambda: os.getenv("SPARK_MASTER", "local[*]"))

    @property
    def bronze_path(self) -> str:
        return f"s3a://{self.bucket}/bronze"

    @property
    def silver_path(self) -> str:
        return f"s3a://{self.bucket}/silver"

    @property
    def gold_path(self) -> str:
        return f"s3a://{self.bucket}/gold"


# ---------------------------------------------------------------------------
# Spark session factory
# ---------------------------------------------------------------------------

def build_spark_session(config: LakehouseConfig) -> SparkSession:
    """Creates a Delta Lake–enabled SparkSession.

    Configures the Spark session with:
    - Delta Lake extensions for ACID semantics and schema evolution.
    - Hadoop S3A connector with AWS credential chain (IAM role on EMR /
      environment variables locally).
    - Glue Data Catalog as Hive Metastore so all tables are registered
      automatically on creation.

    Args:
        config: Lakehouse runtime configuration.

    Returns:
        Configured ``SparkSession`` instance.

    Raises:
        RuntimeError: If Spark session cannot be initialized.
    """
    try:
        builder = (
            SparkSession.builder.appName(f"mlb-catalog-setup-{config.env}")
            .master(config.spark_master)
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            # Use AWS Glue Data Catalog as the Hive metastore
            .config("spark.hadoop.hive.metastore.client.factory.class",
                    "com.amazonaws.glue.catalog.metastore.AWSGlueDataCatalogHiveClientFactory")
            .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                    "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
            # Delta Lake recommended tuning
            .config("spark.databricks.delta.optimizeWrite.enabled", "true")
            .config("spark.databricks.delta.autoCompact.enabled", "true")
            .config("spark.sql.shuffle.partitions", "200")
        )
        session = configure_spark_with_delta_pip(builder).getOrCreate()
        session.sparkContext.setLogLevel(os.getenv("SPARK_LOG_LEVEL", "WARN"))
        log.info("spark_session_created", app_name=session.sparkContext.appName)
        return session
    except Exception as exc:
        log.error("spark_session_failed", error=str(exc))
        raise RuntimeError(f"Failed to initialize SparkSession: {exc}") from exc


# ---------------------------------------------------------------------------
# Glue catalog helpers
# ---------------------------------------------------------------------------

def ensure_glue_database(glue_client, database: str, region: str) -> None:
    """Creates the Glue database if it does not already exist.

    Args:
        glue_client: Boto3 Glue client.
        database: Name of the Glue database to ensure.
        region: AWS region (stored as a database parameter for discoverability).

    Raises:
        ClientError: For any AWS error other than AlreadyExistsException.
    """
    try:
        glue_client.create_database(
            DatabaseInput={
                "Name": database,
                "Description": "MLB Lineup Optimizer Lakehouse — managed by catalog_setup.py",
                "Parameters": {"region": region, "format": "delta"},
            }
        )
        log.info("glue_database_created", database=database)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "AlreadyExistsException":
            log.info("glue_database_exists", database=database)
        else:
            log.error("glue_database_error", error=str(exc))
            raise


def register_delta_table_in_glue(
    glue_client,
    database: str,
    table_name: str,
    s3_location: str,
    schema: StructType,
    partition_keys: list[str],
) -> None:
    """Registers (or updates) a Delta Lake table in AWS Glue Data Catalog.

    Uses ``UpdateTable`` if the table exists, ``CreateTable`` otherwise so
    the function is fully idempotent and safe to re-run in CI pipelines.

    Args:
        glue_client: Boto3 Glue client.
        database: Target Glue database name.
        table_name: Logical table name (e.g. ``plate_appearances``).
        s3_location: S3 path of the Delta table root.
        schema: PySpark ``StructType`` representing the table schema.
        partition_keys: Column names used for partitioning.

    Raises:
        ClientError: For unrecoverable AWS Glue API errors.
    """
    glue_columns = [
        {"Name": f.name, "Type": _spark_type_to_glue(f.dataType)}
        for f in schema.fields
        if f.name not in partition_keys
    ]
    partition_columns = [
        {"Name": f.name, "Type": _spark_type_to_glue(f.dataType)}
        for f in schema.fields
        if f.name in partition_keys
    ]

    table_input = {
        "Name": table_name,
        "Description": f"Delta Lake table managed by catalog_setup.py | {s3_location}",
        "StorageDescriptor": {
            "Columns": glue_columns,
            "Location": s3_location,
            "InputFormat": "org.apache.hadoop.mapred.SequenceFileInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveSequenceFileOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
                "Parameters": {"serialization.format": "1"},
            },
            "Parameters": {"classification": "delta"},
        },
        "PartitionKeys": partition_columns,
        "TableType": "EXTERNAL_TABLE",
        "Parameters": {
            "table_type": "DELTA",
            "spark.sql.sources.provider": "delta",
        },
    }

    try:
        glue_client.get_table(DatabaseName=database, Name=table_name)
        glue_client.update_table(DatabaseName=database, TableInput=table_input)
        log.info("glue_table_updated", table=f"{database}.{table_name}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "EntityNotFoundException":
            glue_client.create_table(DatabaseName=database, TableInput=table_input)
            log.info("glue_table_created", table=f"{database}.{table_name}")
        else:
            log.error("glue_table_error", table=table_name, error=str(exc))
            raise


def _spark_type_to_glue(spark_type) -> str:
    """Maps a PySpark DataType to its Glue / Hive equivalent string.

    Args:
        spark_type: A PySpark ``DataType`` instance.

    Returns:
        Glue-compatible type string.
    """
    mapping = {
        StringType: "string",
        IntegerType: "int",
        LongType: "bigint",
        ShortType: "smallint",
        FloatType: "float",
        DoubleType: "double",
        TimestampType: "timestamp",
        DecimalType: "decimal",
    }
    for spark_cls, glue_type in mapping.items():
        if isinstance(spark_type, spark_cls):
            return glue_type
    return "string"  # safe default for unknown types


# ---------------------------------------------------------------------------
# Delta Lake table creation helpers
# ---------------------------------------------------------------------------

def create_delta_table_if_not_exists(
    spark: SparkSession,
    table_path: str,
    schema: StructType,
    partition_by: list[str],
    table_properties: Optional[dict[str, str]] = None,
) -> None:
    """Creates a Delta Lake table at the given S3 path if it doesn't exist.

    Uses ``DeltaTableBuilder`` (idempotent) so it is safe to call on every
    pipeline run. If the table already exists with a compatible schema, this
    is a no-op.

    Args:
        spark: Active SparkSession with Delta extensions loaded.
        table_path: S3 path where the Delta table root will be created.
        schema: PySpark ``StructType`` for the table.
        partition_by: List of column names to partition on.
        table_properties: Optional Delta table properties dict
            (e.g. ``{"delta.logRetentionDuration": "interval 90 days"}``).

    Raises:
        Exception: Re-raised if Delta table creation fails.
    """
    from delta.tables import DeltaTable  # local import avoids circular dependency at module level

    if DeltaTable.isDeltaTable(spark, table_path):
        log.info("delta_table_exists", path=table_path)
        return

    try:
        props = {
            "delta.autoOptimize.optimizeWrite": "true",
            "delta.autoOptimize.autoCompact": "true",
            "delta.logRetentionDuration": "interval 90 days",
            "delta.deletedFileRetentionDuration": "interval 7 days",
        }
        if table_properties:
            props.update(table_properties)

        builder = (
            DeltaTable.createIfNotExists(spark)
            .location(table_path)
            .addColumns(schema)
            .partitionedBy(*partition_by)
        )
        for key, value in props.items():
            builder = builder.property(key, value)

        builder.execute()
        log.info("delta_table_created", path=table_path, partitions=partition_by)
    except Exception as exc:
        log.error("delta_table_creation_failed", path=table_path, error=str(exc))
        raise


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def setup_lakehouse(config: LakehouseConfig) -> None:
    """Orchestrates the full lakehouse catalog initialization.

    Execution order:
    1. Build Spark session with Delta + Glue integration.
    2. Ensure Glue database exists.
    3. Create Bronze / Silver / Gold Delta tables.
    4. Register each table in Glue Data Catalog.

    Args:
        config: Runtime configuration for bucket, region, etc.

    Raises:
        RuntimeError: If any critical step fails.
    """
    log.info("catalog_setup_start", env=config.env, bucket=config.bucket)

    spark = build_spark_session(config)
    glue = boto3.client("glue", region_name=config.region)

    ensure_glue_database(glue, config.glue_database, config.region)

    # ------------------------------------------------------------------
    # Bronze layer
    # ------------------------------------------------------------------
    bronze_statcast_path = f"{config.bronze_path}/statcast"
    create_delta_table_if_not_exists(
        spark=spark,
        table_path=bronze_statcast_path,
        schema=BRONZE_STATCAST_SCHEMA,
        partition_by=["game_date"],
        table_properties={"delta.logRetentionDuration": "interval 365 days"},
    )
    register_delta_table_in_glue(
        glue_client=glue,
        database=config.glue_database,
        table_name="bronze_statcast",
        s3_location=bronze_statcast_path,
        schema=BRONZE_STATCAST_SCHEMA,
        partition_keys=["game_date"],
    )

    # ------------------------------------------------------------------
    # Silver layer
    # ------------------------------------------------------------------
    silver_pa_path = f"{config.silver_path}/plate_appearances"
    create_delta_table_if_not_exists(
        spark=spark,
        table_path=silver_pa_path,
        schema=SILVER_PLATE_APPEARANCES_SCHEMA,
        partition_by=["game_date", "venue_id"],
        table_properties={
            "delta.logRetentionDuration": "interval 3650 days",  # 10 years
            "delta.enableChangeDataFeed": "true",                # CDC for Feature Store sync
        },
    )
    register_delta_table_in_glue(
        glue_client=glue,
        database=config.glue_database,
        table_name="silver_plate_appearances",
        s3_location=silver_pa_path,
        schema=SILVER_PLATE_APPEARANCES_SCHEMA,
        partition_keys=["game_date", "venue_id"],
    )

    # ------------------------------------------------------------------
    # Gold layer
    # ------------------------------------------------------------------
    gold_batter_path = f"{config.gold_path}/batter_features_rolling"
    create_delta_table_if_not_exists(
        spark=spark,
        table_path=gold_batter_path,
        schema=GOLD_BATTER_FEATURES_SCHEMA,
        partition_by=["feature_date"],
        table_properties={"delta.logRetentionDuration": "interval 1825 days"},  # 5 years
    )
    register_delta_table_in_glue(
        glue_client=glue,
        database=config.glue_database,
        table_name="gold_batter_features_rolling",
        s3_location=gold_batter_path,
        schema=GOLD_BATTER_FEATURES_SCHEMA,
        partition_keys=["feature_date"],
    )

    log.info("catalog_setup_complete", tables_registered=3)
    spark.stop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize the MLB Lineup Optimizer Lakehouse catalog."
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name (without s3://)")
    parser.add_argument("--glue-database", default="mlb_lakehouse", dest="glue_database")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--env", choices=["dev", "staging", "prod"], default="dev")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for catalog initialization.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]`` when ``None``.
    """
    args = _parse_args(argv)
    config = LakehouseConfig(
        bucket=args.bucket,
        glue_database=args.glue_database,
        region=args.region,
        env=args.env,
    )
    setup_lakehouse(config)


if __name__ == "__main__":
    main()
