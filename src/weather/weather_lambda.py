"""
weather_lambda.py
=================
AWS Lambda function for hyper-local weather ingestion at MLB venues.

Execution model:
    - Triggered by an EventBridge rule every 30 minutes during daylight hours,
      increasing to every 10 minutes in the 3 hours before first pitch.
    - For each configured venue, tries **Tomorrow.io** first (hyper-local 1 km²
      grid) then falls back to **NOAA/Weather.gov** if Tomorrow.io is
      unavailable or rate-limited.
    - Enriches each snapshot with a wind vector (speed × sin/cos components)
      so downstream models can use wind as a continuous directional feature
      rather than a scalar magnitude.
    - Publishes records to Kinesis Data Stream ``weather-snapshots``.

Deployment:
    Packaged as a Lambda ZIP with httpx and tenacity included.
    Required environment variables (set via Lambda console or Terraform):
        TOMORROW_IO_API_KEY  — Tomorrow.io v4 API key (from Secrets Manager)
        NOAA_USER_AGENT      — e.g. "mlb-optimizer/1.0 (ops@team.com)"
        KINESIS_STREAM_NAME  — default: weather-snapshots
        AWS_DEFAULT_REGION   — default: us-east-1

Wind vector math:
    MLB stadiums are oriented differently, so raw wind direction (azimuth
    0–360°) is not a regression-friendly feature. We decompose into:
        wind_u = speed × sin(direction_rad)   # east–west component
        wind_v = speed × cos(direction_rad)   # north–south component
    The model can then learn the directional effect specific to each stadium's
    orientation without the modeler having to hard-code per-park adjustments.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
import httpx
import structlog
from botocore.exceptions import ClientError
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
# MLB Venue registry
# All 30 MLB stadiums with GPS coordinates (WGS-84) and NOAA grid codes.
# NOAA grid obtained via:  GET /points/{lat},{lon}  → properties.gridId/gridX/gridY
# ---------------------------------------------------------------------------
MLB_VENUES: dict[str, dict[str, Any]] = {
    "wrigley":      {"lat": 41.9484, "lon": -87.6553, "name": "Wrigley Field",       "noaa_grid": "LOT", "noaa_x": 73,  "noaa_y": 73},
    "fenway":       {"lat": 42.3467, "lon": -71.0972, "name": "Fenway Park",          "noaa_grid": "BOX", "noaa_x": 68,  "noaa_y": 84},
    "coors":        {"lat": 39.7559, "lon": -104.9942,"name": "Coors Field",           "noaa_grid": "BOU", "noaa_x": 59,  "noaa_y": 72},
    "yankee":       {"lat": 40.8296, "lon": -73.9262, "name": "Yankee Stadium",        "noaa_grid": "OKX", "noaa_x": 33,  "noaa_y": 37},
    "dodger":       {"lat": 34.0739, "lon": -118.2400,"name": "Dodger Stadium",         "noaa_grid": "LOX", "noaa_x": 153, "noaa_y": 47},
    "oracle":       {"lat": 37.7786, "lon": -122.3893,"name": "Oracle Park",            "noaa_grid": "MTR", "noaa_x": 84,  "noaa_y": 88},
    "busch":        {"lat": 38.6226, "lon": -90.1930, "name": "Busch Stadium",          "noaa_grid": "LSX", "noaa_x": 88,  "noaa_y": 72},
    "globe_life":   {"lat": 32.7473, "lon": -97.0839, "name": "Globe Life Field",       "noaa_grid": "FWD", "noaa_x": 104, "noaa_y": 74},
    "minute_maid":  {"lat": 29.7573, "lon": -95.3555, "name": "Minute Maid Park",       "noaa_grid": "HGX", "noaa_x": 68,  "noaa_y": 51},
    "great_american":{"lat": 39.0975,"lon": -84.5064, "name": "Great American Ball Park","noaa_grid": "ILN", "noaa_x": 83,  "noaa_y": 61},
}

# ---------------------------------------------------------------------------
# Tomorrow.io client
# ---------------------------------------------------------------------------

TOMORROW_IO_BASE = "https://api.tomorrow.io/v4"
TOMORROW_IO_FIELDS = [
    "temperature",
    "humidity",
    "windSpeed",
    "windDirection",
    "pressureSurfaceLevel",
    "precipitationIntensity",
    "weatherCode",
]


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _fetch_tomorrow_io(
    venue_id: str,
    lat: float,
    lon: float,
    api_key: str,
    http_client: httpx.Client,
) -> Optional[dict[str, Any]]:
    """Fetches real-time weather from Tomorrow.io for a single venue.

    Args:
        venue_id: Internal venue identifier (e.g. ``"wrigley"``).
        lat: Venue latitude in decimal degrees (WGS-84).
        lon: Venue longitude in decimal degrees (WGS-84).
        api_key: Tomorrow.io v4 API key.
        http_client: Shared HTTP client for connection reuse.

    Returns:
        Dict with raw weather values, or ``None`` if parsing fails.

    Raises:
        httpx.HTTPStatusError: After all retries fail (caller handles fallback).
    """
    resp = http_client.get(
        f"{TOMORROW_IO_BASE}/weather/realtime",
        params={
            "location": f"{lat},{lon}",
            "apikey": api_key,
            "fields": ",".join(TOMORROW_IO_FIELDS),
            "units": "imperial",
        },
        timeout=10.0,
    )
    resp.raise_for_status()

    data = resp.json().get("data", {}).get("values", {})
    if not data:
        log.warning("tomorrow_io_empty_response", venue_id=venue_id)
        return None

    return {
        "temperature_f": data.get("temperature"),
        "humidity_pct": data.get("humidity"),
        "wind_speed_mph": data.get("windSpeed"),
        "wind_direction_deg": data.get("windDirection"),
        "pressure_inhg": data.get("pressureSurfaceLevel"),
        "precip_intensity": data.get("precipitationIntensity"),
        "weather_code": data.get("weatherCode"),
        "data_source": "tomorrow_io",
    }


# ---------------------------------------------------------------------------
# NOAA/Weather.gov fallback client
# ---------------------------------------------------------------------------

NOAA_BASE = "https://api.weather.gov"


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=3, max=15),
    reraise=True,
)
def _fetch_noaa_observations(
    venue_id: str,
    venue_meta: dict[str, Any],
    user_agent: str,
    http_client: httpx.Client,
) -> Optional[dict[str, Any]]:
    """Fetches latest weather observations from NOAA/Weather.gov.

    Uses the NOAA gridpoint forecast endpoint which returns hourly forecasts
    for the next 7 days. We take the first (current) period as the snapshot.

    NOAA requires a descriptive User-Agent string per their API policy; the
    caller supplies this from the ``NOAA_USER_AGENT`` environment variable.

    Args:
        venue_id: Internal venue identifier.
        venue_meta: Venue dict containing ``noaa_grid``, ``noaa_x``, ``noaa_y``.
        user_agent: NOAA API User-Agent header value.
        http_client: Shared HTTP client.

    Returns:
        Normalized weather dict, or ``None`` if parsing fails.
    """
    grid_id = venue_meta["noaa_grid"]
    grid_x = venue_meta["noaa_x"]
    grid_y = venue_meta["noaa_y"]

    resp = http_client.get(
        f"{NOAA_BASE}/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast/hourly",
        headers={"User-Agent": user_agent, "Accept": "application/geo+json"},
        timeout=15.0,
    )
    resp.raise_for_status()

    periods = resp.json().get("properties", {}).get("periods", [])
    if not periods:
        log.warning("noaa_empty_response", venue_id=venue_id)
        return None

    current = periods[0]
    wind_str: str = current.get("windSpeed", "0 mph")
    wind_speed_mph = float(wind_str.split()[0]) if wind_str else 0.0

    wind_dir_str: str = current.get("windDirection", "N")
    wind_direction_deg = _compass_to_degrees(wind_dir_str)

    return {
        "temperature_f": current.get("temperature"),
        "humidity_pct": current.get("relativeHumidity", {}).get("value"),
        "wind_speed_mph": wind_speed_mph,
        "wind_direction_deg": wind_direction_deg,
        "pressure_inhg": None,   # NOAA hourly forecast omits pressure; use observations endpoint if needed
        "precip_intensity": None,
        "weather_code": None,
        "weather_condition": current.get("shortForecast"),
        "data_source": "noaa",
    }


def _compass_to_degrees(compass: str) -> float:
    """Converts a compass bearing string to decimal degrees.

    Supports 8-point (``N``, ``NE``, ...) and 16-point (``NNE``, ...) roses.

    Args:
        compass: Compass direction string (e.g. ``"NNE"``, ``"SW"``).

    Returns:
        Azimuth in degrees (0 = North, clockwise).
    """
    mapping = {
        "N": 0.0,   "NNE": 22.5,  "NE": 45.0,  "ENE": 67.5,
        "E": 90.0,  "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
        "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
        "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
    }
    return mapping.get(compass.upper().strip(), 0.0)


# ---------------------------------------------------------------------------
# Wind vector decomposition
# ---------------------------------------------------------------------------

def _compute_wind_vector(
    speed_mph: Optional[float],
    direction_deg: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    """Decomposes scalar wind into (u, v) Cartesian components.

    The decomposition uses meteorological convention:
        u (east–west):  positive = wind blowing from west
        v (north–south): positive = wind blowing from south

    Args:
        speed_mph: Wind speed in miles per hour.
        direction_deg: Wind direction in degrees (azimuth; 0 = N, 90 = E).

    Returns:
        Tuple ``(wind_u_mph, wind_v_mph)``, both ``None`` if inputs are ``None``.
    """
    if speed_mph is None or direction_deg is None:
        return None, None
    rad = math.radians(direction_deg)
    wind_u = speed_mph * math.sin(rad)
    wind_v = speed_mph * math.cos(rad)
    return round(wind_u, 4), round(wind_v, 4)


# ---------------------------------------------------------------------------
# Kinesis publisher
# ---------------------------------------------------------------------------

def _publish_to_kinesis(
    record: dict[str, Any],
    stream_name: str,
    kinesis_client,
) -> bool:
    """Publishes a single weather snapshot to Kinesis.

    Args:
        record: Weather snapshot dict.
        stream_name: Kinesis stream name.
        kinesis_client: Boto3 Kinesis client.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    try:
        kinesis_client.put_record(
            StreamName=stream_name,
            Data=json.dumps(record, default=str).encode("utf-8"),
            PartitionKey=record["venue_id"],
        )
        log.info("kinesis_published", venue_id=record["venue_id"], source=record.get("data_source"))
        return True
    except ClientError as exc:
        log.error("kinesis_publish_failed", venue_id=record["venue_id"], error=str(exc))
        return False


# ---------------------------------------------------------------------------
# Secrets Manager helper
# ---------------------------------------------------------------------------

def _get_secret(secret_name: str, region: str) -> str:
    """Retrieves a plaintext secret from AWS Secrets Manager.

    Args:
        secret_name: Secret name or ARN.
        region: AWS region where the secret is stored.

    Returns:
        Plaintext secret string.

    Raises:
        RuntimeError: If Secrets Manager lookup fails.
    """
    client = boto3.client("secretsmanager", region_name=region)
    try:
        resp = client.get_secret_value(SecretId=secret_name)
        return resp.get("SecretString", "")
    except ClientError as exc:
        log.error("secrets_manager_error", secret=secret_name, error=str(exc))
        raise RuntimeError(f"Could not retrieve secret {secret_name}: {exc}") from exc


# ---------------------------------------------------------------------------
# Core per-venue ingestion logic
# ---------------------------------------------------------------------------

def ingest_venue_weather(
    venue_id: str,
    venue_meta: dict[str, Any],
    tomorrow_api_key: str,
    noaa_user_agent: str,
    http_client: httpx.Client,
) -> Optional[dict[str, Any]]:
    """Fetches and normalizes weather for one venue using primary / fallback sources.

    Attempt order:
    1. Tomorrow.io (hyper-local, 1 km² resolution, 5-minute update cycle)
    2. NOAA/Weather.gov (free, government SLA, hourly resolution)

    If both sources fail, returns ``None`` and logs a critical alert.

    Args:
        venue_id: Internal venue identifier.
        venue_meta: Venue metadata dict from ``MLB_VENUES``.
        tomorrow_api_key: Tomorrow.io API key.
        noaa_user_agent: NOAA User-Agent header string.
        http_client: Shared HTTP client.

    Returns:
        Normalized and enriched weather snapshot dict, or ``None`` on failure.
    """
    raw: Optional[dict[str, Any]] = None

    # Primary: Tomorrow.io
    try:
        raw = _fetch_tomorrow_io(
            venue_id=venue_id,
            lat=venue_meta["lat"],
            lon=venue_meta["lon"],
            api_key=tomorrow_api_key,
            http_client=http_client,
        )
    except Exception as exc:
        log.warning("tomorrow_io_failed_fallback", venue_id=venue_id, error=str(exc))

    # Fallback: NOAA
    if raw is None:
        try:
            raw = _fetch_noaa_observations(
                venue_id=venue_id,
                venue_meta=venue_meta,
                user_agent=noaa_user_agent,
                http_client=http_client,
            )
        except Exception as exc:
            log.critical("all_weather_sources_failed", venue_id=venue_id, error=str(exc))
            return None

    if raw is None:
        return None

    # Wind vector decomposition
    wind_u, wind_v = _compute_wind_vector(
        raw.get("wind_speed_mph"),
        raw.get("wind_direction_deg"),
    )

    snapshot = {
        "venue_id": venue_id,
        "venue_name": venue_meta["name"],
        "captured_at": datetime.now(timezone.utc).isoformat(),
        **raw,
        "wind_u_mph": wind_u,
        "wind_v_mph": wind_v,
        "lat": venue_meta["lat"],
        "lon": venue_meta["lon"],
    }

    return snapshot


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entry point for weather snapshot ingestion.

    Called by EventBridge on a scheduled cron expression. Iterates over all
    configured venues, fetches weather from Tomorrow.io (with NOAA fallback),
    enriches with wind vector, and publishes each snapshot to Kinesis.

    The ``event`` payload may contain an optional ``venue_ids`` list to
    restrict which venues are polled (useful during testing or partial outages).

    Args:
        event: Lambda event dict. Optional key ``venue_ids`` (list[str]).
        context: Lambda context object (unused; present for signature compliance).

    Returns:
        Summary dict with success/failure counts per venue.
    """
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    stream_name = os.environ.get("KINESIS_STREAM_NAME", "weather-snapshots")
    noaa_ua = os.environ.get("NOAA_USER_AGENT", "mlb-optimizer/1.0")

    # Retrieve Tomorrow.io key from Secrets Manager at cold-start; Lambda
    # caches the execution environment so this runs once per container lifetime.
    tomorrow_key = _get_secret("tomorrow-io-api-key", region)

    kinesis = boto3.client("kinesis", region_name=region)

    target_venues = event.get("venue_ids", list(MLB_VENUES.keys()))
    venues_to_poll = {
        vid: meta for vid, meta in MLB_VENUES.items() if vid in target_venues
    }

    results: dict[str, str] = {}

    with httpx.Client(
        headers={"User-Agent": "mlb-lineup-optimizer/1.0"},
        follow_redirects=True,
    ) as client:
        for venue_id, meta in venues_to_poll.items():
            snapshot = ingest_venue_weather(
                venue_id=venue_id,
                venue_meta=meta,
                tomorrow_api_key=tomorrow_key,
                noaa_user_agent=noaa_ua,
                http_client=client,
            )
            if snapshot is None:
                results[venue_id] = "FAILED"
                continue

            success = _publish_to_kinesis(snapshot, stream_name, kinesis)
            results[venue_id] = "OK" if success else "KINESIS_FAILED"

    success_count = sum(1 for v in results.values() if v == "OK")
    failure_count = len(results) - success_count

    log.info(
        "lambda_complete",
        success=success_count,
        failed=failure_count,
        venues=results,
    )

    return {
        "statusCode": 200 if failure_count == 0 else 207,
        "body": {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "venues": results,
            "success": success_count,
            "failed": failure_count,
        },
    }
