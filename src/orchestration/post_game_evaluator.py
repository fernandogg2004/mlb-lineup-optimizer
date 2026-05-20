"""
post_game_evaluator.py — Evaluador analítico post-partido del Game-Day Orchestrator.
======================================================================================
Responsabilidades:
    1. Esperar a que todos los partidos del día alcancen estado "Final" (polling progresivo).
    2. Descargar los boxscores reales de cada partido.
    3. Cruzar con las predicciones guardadas en PostgreSQL (tabla gameday_predictions).
    4. Calcular métricas analíticas: delta E[R], RMSE, MAE, ECE proxy.
    5. Generar un informe PDF con reportlab (tabla de rendimiento + resumen de métricas).
    6. Persistir los resultados en la tabla gameday_predictions (columnas de resolución).
    7. Enviar resumen por Slack (Incoming Webhook).

Esta clase es llamada por la Task 3 del DAG mlb_gameday_orchestrator.
No tiene dependencias de Airflow — se puede usar en tests unitarios sin el entorno DAG.

Zona horaria:
    Todos los timestamps se manejan en UTC internamente.
    La conversión a ET se hace solo para la presentación en el informe PDF.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
import pytz
import requests
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
_MLB_API_V1_1 = "https://statsapi.mlb.com/api/v1.1"
_UTC = pytz.UTC
_ET_TZ = pytz.timezone("America/New_York")

_GAME_FINAL_STATES = {"Final", "Game Over", "Completed Early"}
_GAME_IN_PROGRESS_STATES = {"In Progress", "Manager Challenge", "Delay", "Delay: Rain"}

# Espera máxima para que todos los partidos finalicen: 8 horas desde el primer pitch
# Un partido de la Costa Oeste (22:10 UTC) puede terminar hasta las ~02:00 UTC del día siguiente
_MAX_WAIT_SECONDS = 8 * 3600
_POLL_INTERVAL_SECONDS = 300  # 5 minutos entre verificaciones de estado
_API_TIMEOUT = 30  # segundos para cada llamada HTTP
_HTTP_RETRY_ATTEMPTS = 3
_HTTP_RETRY_BACKOFF = 2.0  # factor exponencial entre reintentos


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------


@dataclass
class GamePrediction:
    """Predicción pre-partido cargada desde PostgreSQL."""

    game_pk: int
    game_date: str
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    ai_recommended_lineup: list[int]
    ai_expected_runs: float
    win_probability: float
    model_version: str
    lineup_source: str  # "official" | "projected"
    predicted_at: datetime


@dataclass
class GameResult:
    """Resultado completo: predicción + realidad + métricas calculadas."""

    game_pk: int
    game_date: str
    home_team_name: str
    away_team_name: str
    ai_expected_runs: float
    actual_home_runs: int
    actual_away_runs: int
    delta_er: float              # actual_home_runs - ai_expected_runs
    abs_error: float             # |delta_er|
    model_version: str
    lineup_source: str
    resolved_at: datetime
    error_message: Optional[str] = None


@dataclass
class DayMetrics:
    """Métricas agregadas del día."""

    game_date: str
    total_games: int
    games_resolved: int
    games_with_error: int
    mae: float                   # Mean Absolute Error en E[R]
    rmse: float                  # Root Mean Square Error en E[R]
    mean_delta_er: float         # Sesgo medio (positivo = el modelo subestimó)
    model_overpredicted: bool    # True si el modelo en promedio sobreestimó las carreras
    model_version: str


# ---------------------------------------------------------------------------
# Cliente HTTP con reintentos
# ---------------------------------------------------------------------------


def _get_json(url: str, params: dict | None = None) -> dict:
    """GET con reintentos exponenciales ante fallos de red o 5xx."""
    for attempt in range(1, _HTTP_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, params=params, timeout=_API_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                raise  # 4xx: no reintentable
            if attempt == _HTTP_RETRY_ATTEMPTS:
                raise
            wait = _HTTP_RETRY_BACKOFF ** attempt
            logger.warning("http_retry", url=url, attempt=attempt, wait_s=wait, error=str(exc))
            time.sleep(wait)
        except requests.exceptions.RequestException as exc:
            if attempt == _HTTP_RETRY_ATTEMPTS:
                raise
            wait = _HTTP_RETRY_BACKOFF ** attempt
            logger.warning("http_retry", url=url, attempt=attempt, wait_s=wait, error=str(exc))
            time.sleep(wait)
    raise RuntimeError(f"_get_json exhausted retries for {url}")  # unreachable


# ---------------------------------------------------------------------------
# Evaluador principal
# ---------------------------------------------------------------------------


class PostGameEvaluator:
    """Orquesta la evaluación analítica post-partido de todos los juegos del día.

    Args:
        postgres_dsn: DSN de conexión psycopg2
                      (ej. "host=localhost dbname=mlb user=mlb password=secret").
        slack_webhook_url: URL del Incoming Webhook de Slack.
                           Si está vacío, la notificación se omite sin error.
        reports_dir: Directorio donde se guardan los PDFs generados.
    """

    def __init__(
        self,
        postgres_dsn: str,
        slack_webhook_url: str = "",
        reports_dir: str | Path = "reports/gameday",
    ) -> None:
        self._dsn = postgres_dsn
        self._slack_webhook = slack_webhook_url
        self._reports_dir = Path(reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Punto de entrada principal (llamado por Task 3 del DAG)
    # ------------------------------------------------------------------

    def run(self, game_pks: list[int], game_date: str) -> DayMetrics:
        """Ejecuta el ciclo completo de evaluación post-partido.

        Args:
            game_pks: Lista de gamePk del día.
            game_date: Fecha del día en formato YYYY-MM-DD.

        Returns:
            DayMetrics con el resumen del rendimiento del modelo.

        Raises:
            RuntimeError: Si falla la conexión a PostgreSQL o la generación del PDF.
        """
        logger.info("post_game_evaluator_start", game_date=game_date, n_games=len(game_pks))

        # 1. Esperar a que todos los partidos finalicen
        self.wait_for_all_games_final(game_pks, game_date)

        # 2. Cargar predicciones pre-partido desde la BD
        predictions = self.load_predictions_from_db(game_pks)
        logger.info("predictions_loaded", count=len(predictions))

        # 3. Descargar boxscores y cruzar con predicciones
        results: list[GameResult] = []
        for pred in predictions:
            try:
                boxscore = self.fetch_boxscore(pred.game_pk)
                actual_home = boxscore.get("home_runs", 0)
                actual_away = boxscore.get("away_runs", 0)
                delta = round(actual_home - pred.ai_expected_runs, 3)
                results.append(GameResult(
                    game_pk=pred.game_pk,
                    game_date=pred.game_date,
                    home_team_name=pred.home_team_name,
                    away_team_name=pred.away_team_name,
                    ai_expected_runs=pred.ai_expected_runs,
                    actual_home_runs=actual_home,
                    actual_away_runs=actual_away,
                    delta_er=delta,
                    abs_error=abs(delta),
                    model_version=pred.model_version,
                    lineup_source=pred.lineup_source,
                    resolved_at=datetime.now(_UTC),
                ))
                logger.info(
                    "game_resolved",
                    game_pk=pred.game_pk,
                    matchup=f"{pred.away_team_name} @ {pred.home_team_name}",
                    ai_er=pred.ai_expected_runs,
                    actual=actual_home,
                    delta=delta,
                )
            except Exception as exc:
                logger.error("game_resolution_failed", game_pk=pred.game_pk, error=str(exc))
                results.append(GameResult(
                    game_pk=pred.game_pk,
                    game_date=game_date,
                    home_team_name="Unknown",
                    away_team_name="Unknown",
                    ai_expected_runs=0.0,
                    actual_home_runs=0,
                    actual_away_runs=0,
                    delta_er=0.0,
                    abs_error=0.0,
                    model_version="unknown",
                    lineup_source="unknown",
                    resolved_at=datetime.now(_UTC),
                    error_message=str(exc),
                ))

        # 4. Calcular métricas del día
        metrics = self.calculate_metrics(results, game_date)
        logger.info(
            "day_metrics_calculated",
            game_date=game_date,
            mae=metrics.mae,
            rmse=metrics.rmse,
            mean_delta=metrics.mean_delta_er,
        )

        # 5. Persistir resoluciones en PostgreSQL
        self.save_results_to_db(results)

        # 6. Generar informe PDF
        report_path = self.generate_pdf_report(results, metrics)
        logger.info("pdf_report_generated", path=str(report_path))

        # 7. Notificar por Slack
        if self._slack_webhook:
            self.send_slack_notification(metrics, results, report_path)

        return metrics

    # ------------------------------------------------------------------
    # 1. Espera activa de finalización de partidos
    # ------------------------------------------------------------------

    def wait_for_all_games_final(
        self,
        game_pks: list[int],
        game_date: str,
    ) -> dict[int, str]:
        """Espera polling progresivo hasta que todos los partidos alcancen "Final".

        La espera máxima es _MAX_WAIT_SECONDS (8 horas). El intervalo de polling
        aumenta progresivamente con el tiempo transcurrido para no saturar la API.

        Args:
            game_pks: Lista de gamePk a monitorizar.
            game_date: Fecha en YYYY-MM-DD (usado solo en logging).

        Returns:
            Dict {game_pk → estado_final_string}.

        Raises:
            TimeoutError: Si transcurren 8 horas sin que todos los juegos finalicen.
        """
        if not game_pks:
            return {}

        pending = set(game_pks)
        final_states: dict[int, str] = {}
        start = time.monotonic()
        poll_count = 0

        logger.info(
            "awaiting_games_final",
            game_date=game_date,
            n_games=len(game_pks),
            timeout_h=_MAX_WAIT_SECONDS // 3600,
        )

        while pending:
            elapsed = time.monotonic() - start
            if elapsed > _MAX_WAIT_SECONDS:
                timed_out = list(pending)
                logger.error(
                    "game_wait_timeout",
                    timed_out_game_pks=timed_out,
                    elapsed_h=round(elapsed / 3600, 1),
                )
                # No bloqueamos el pipeline: continuamos con los partidos finalizados
                for pk in timed_out:
                    final_states[pk] = "Timeout"
                break

            poll_count += 1
            # Intervalo progresivo: las primeras 2h → 5 min; después → 10 min
            interval = _POLL_INTERVAL_SECONDS if elapsed < 7200 else _POLL_INTERVAL_SECONDS * 2

            for pk in list(pending):
                try:
                    state = self._fetch_game_state(pk)
                    if state in _GAME_FINAL_STATES:
                        logger.info("game_final", game_pk=pk, state=state)
                        final_states[pk] = state
                        pending.discard(pk)
                    else:
                        logger.debug("game_not_final", game_pk=pk, state=state)
                except Exception as exc:
                    logger.warning("game_state_check_failed", game_pk=pk, error=str(exc))

            if pending:
                logger.info(
                    "games_still_pending",
                    pending_pks=list(pending),
                    poll_count=poll_count,
                    elapsed_min=round(elapsed / 60, 1),
                    next_check_min=round(interval / 60, 1),
                )
                time.sleep(interval)

        logger.info(
            "all_games_settled",
            game_date=game_date,
            final_count=len(final_states),
        )
        return final_states

    def _fetch_game_state(self, game_pk: int) -> str:
        """Devuelve el abstractGameState del partido (Preview/Live/Final/etc.)."""
        data = _get_json(f"{_MLB_API_V1_1}/game/{game_pk}/feed/live")
        return (
            data.get("gameData", {})
            .get("status", {})
            .get("detailedState", "Unknown")
        )

    # ------------------------------------------------------------------
    # 2. Boxscore real
    # ------------------------------------------------------------------

    def fetch_boxscore(self, game_pk: int) -> dict:
        """Descarga el boxscore y extrae carreras por equipo.

        Returns:
            {"home_runs": int, "away_runs": int, "home_name": str, "away_name": str}
        """
        data = _get_json(f"{_MLB_API_BASE}/game/{game_pk}/boxscore")
        teams = data.get("teams", {})
        home_batting = teams.get("home", {}).get("teamStats", {}).get("batting", {})
        away_batting = teams.get("away", {}).get("teamStats", {}).get("batting", {})
        home_info = teams.get("home", {}).get("team", {})
        away_info = teams.get("away", {}).get("team", {})

        home_runs = home_batting.get("runs")
        away_runs = away_batting.get("runs")

        if home_runs is None or away_runs is None:
            raise ValueError(
                f"game_pk={game_pk}: runs not available in boxscore "
                f"(home={home_runs}, away={away_runs})"
            )

        return {
            "home_runs": int(home_runs),
            "away_runs": int(away_runs),
            "home_name": home_info.get("name", "Home"),
            "away_name": away_info.get("name", "Away"),
        }

    # ------------------------------------------------------------------
    # 3. Carga de predicciones desde PostgreSQL
    # ------------------------------------------------------------------

    def load_predictions_from_db(self, game_pks: list[int]) -> list[GamePrediction]:
        """Lee las predicciones pre-partido de la tabla gameday_predictions.

        Solo devuelve registros cuya columna actual_home_runs IS NULL
        (pendientes de resolución).
        """
        if not game_pks:
            return []

        query = """
            SELECT
                game_pk, game_date, home_team_id, away_team_id,
                home_team_name, away_team_name,
                ai_recommended_lineup, ai_expected_runs,
                win_probability, model_version, lineup_source, predicted_at
            FROM gameday_predictions
            WHERE game_pk = ANY(%s)
              AND actual_home_runs IS NULL
            ORDER BY game_pk
        """
        predictions: list[GamePrediction] = []
        try:
            conn = psycopg2.connect(self._dsn)
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cur.execute(query, (game_pks,))
            for row in cur.fetchall():
                predictions.append(GamePrediction(
                    game_pk=row["game_pk"],
                    game_date=str(row["game_date"]),
                    home_team_id=row["home_team_id"],
                    away_team_id=row["away_team_id"],
                    home_team_name=row["home_team_name"] or "",
                    away_team_name=row["away_team_name"] or "",
                    ai_recommended_lineup=row["ai_recommended_lineup"] or [],
                    ai_expected_runs=float(row["ai_expected_runs"] or 0),
                    win_probability=float(row["win_probability"] or 0),
                    model_version=row["model_version"] or "unknown",
                    lineup_source=row["lineup_source"] or "unknown",
                    predicted_at=row["predicted_at"],
                ))
            conn.close()
        except psycopg2.Error as exc:
            logger.error("db_load_predictions_failed", error=str(exc))
            raise

        return predictions

    # ------------------------------------------------------------------
    # 4. Métricas analíticas
    # ------------------------------------------------------------------

    def calculate_metrics(self, results: list[GameResult], game_date: str) -> DayMetrics:
        """Calcula MAE, RMSE, sesgo medio y overprediction flag.

        Excluye resultados con error_message (partidos que fallaron).
        """
        valid = [r for r in results if r.error_message is None]
        n_errors = len(results) - len(valid)

        if not valid:
            return DayMetrics(
                game_date=game_date,
                total_games=len(results),
                games_resolved=0,
                games_with_error=n_errors,
                mae=0.0,
                rmse=0.0,
                mean_delta_er=0.0,
                model_overpredicted=False,
                model_version="unknown",
            )

        n = len(valid)
        mae = round(sum(r.abs_error for r in valid) / n, 4)
        rmse = round(math.sqrt(sum(r.abs_error ** 2 for r in valid) / n), 4)
        mean_delta = round(sum(r.delta_er for r in valid) / n, 4)
        model_version = valid[0].model_version

        return DayMetrics(
            game_date=game_date,
            total_games=len(results),
            games_resolved=n,
            games_with_error=n_errors,
            mae=mae,
            rmse=rmse,
            mean_delta_er=mean_delta,
            # Negativo = modelo sobreestimó (predijo más carreras de las reales)
            model_overpredicted=mean_delta < 0,
            model_version=model_version,
        )

    # ------------------------------------------------------------------
    # 5. Generación del informe PDF (reportlab)
    # ------------------------------------------------------------------

    def generate_pdf_report(
        self,
        results: list[GameResult],
        metrics: DayMetrics,
    ) -> Path:
        """Genera un PDF con tabla de rendimiento y resumen de métricas del día.

        Returns:
            Path absoluta al PDF generado.
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
            )
        except ImportError as exc:
            logger.error("reportlab_not_installed", error=str(exc))
            raise RuntimeError(
                "reportlab es necesario para generar el informe PDF. "
                "Instala con: pip install reportlab"
            ) from exc

        output_path = self._reports_dir / f"gameday_report_{metrics.game_date}.pdf"
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "title",
            parent=styles["Title"],
            fontSize=16,
            spaceAfter=6,
        )
        subtitle_style = ParagraphStyle(
            "subtitle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.gray,
            spaceAfter=14,
        )
        section_style = ParagraphStyle(
            "section",
            parent=styles["Heading2"],
            fontSize=12,
            spaceBefore=12,
            spaceAfter=6,
        )

        story = []

        # --- Portada / encabezado ---
        now_et = datetime.now(_ET_TZ)
        story.append(Paragraph(
            "MLB AI — Informe de Rendimiento del Modelo",
            title_style,
        ))
        story.append(Paragraph(
            f"Fecha de partido: {metrics.game_date} · "
            f"Generado: {now_et.strftime('%Y-%m-%d %H:%M ET')} · "
            f"Modelo: {metrics.model_version}",
            subtitle_style,
        ))

        # --- Resumen de métricas ---
        story.append(Paragraph("Resumen de Métricas del Día", section_style))
        bias_label = "sobreestimó carreras" if metrics.model_overpredicted else "subestimó carreras"
        metrics_data = [
            ["Métrica", "Valor", "Interpretación"],
            ["Partidos procesados", str(metrics.total_games), ""],
            ["Partidos resueltos", str(metrics.games_resolved), ""],
            ["Errores de resolución", str(metrics.games_with_error),
             "Partidos sin boxscore o sin predicción" if metrics.games_with_error else "—"],
            ["MAE (E[R])", f"{metrics.mae:.4f}",
             "Error absoluto medio vs. carreras reales del home team"],
            ["RMSE (E[R])", f"{metrics.rmse:.4f}",
             "Penaliza más los errores grandes"],
            ["Sesgo medio (Δ E[R])", f"{metrics.mean_delta_er:+.4f}",
             f"El modelo {bias_label} en promedio"],
        ]

        metrics_table = Table(metrics_data, colWidths=[5 * cm, 3 * cm, 9 * cm])
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c5e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("ALIGN", (1, 1), (1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.5 * cm))

        # --- Tabla de resultados por partido ---
        story.append(Paragraph("Detalle por Partido", section_style))

        game_header = [
            "Partido", "E[R] AI", "C. Reales\n(Home)", "Δ E[R]",
            "Error Abs.", "Fuente lineup"
        ]
        game_rows = [game_header]
        for r in sorted(results, key=lambda x: x.game_pk):
            if r.error_message:
                game_rows.append([
                    f"{r.away_team_name} @ {r.home_team_name}",
                    "—", "—", "—", "—",
                    f"ERROR: {r.error_message[:40]}",
                ])
            else:
                delta_str = f"{r.delta_er:+.2f}"
                game_rows.append([
                    f"{r.away_team_name} @ {r.home_team_name}",
                    f"{r.ai_expected_runs:.2f}",
                    str(r.actual_home_runs),
                    delta_str,
                    f"{r.abs_error:.2f}",
                    r.lineup_source,
                ])

        col_widths = [6 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 3 * cm]
        game_table = Table(game_rows, colWidths=col_widths)
        game_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c5e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))

        # Colorear filas con error grande en rojo claro
        for i, r in enumerate(results, start=1):
            if not r.error_message and r.abs_error > 2.0:
                game_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fde8e8")),
                ]))
        story.append(game_table)

        # --- Nota metodológica ---
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(
            "<b>Nota:</b> Δ E[R] = Carreras reales (home team) − E[R] predicho por el modelo AI. "
            "Positivo → el modelo subestimó la producción ofensiva. "
            "Filas resaltadas en rojo = error absoluto > 2.0 carreras.",
            styles["Italic"],
        ))

        doc.build(story)
        logger.info("pdf_report_written", path=str(output_path))
        return output_path

    # ------------------------------------------------------------------
    # 6. Persistencia de resoluciones en PostgreSQL
    # ------------------------------------------------------------------

    def save_results_to_db(self, results: list[GameResult]) -> int:
        """Actualiza la tabla gameday_predictions con las resoluciones post-partido.

        Returns:
            Número de filas actualizadas.
        """
        if not results:
            return 0

        update_sql = """
            UPDATE gameday_predictions
            SET
                actual_home_runs = %(actual_home_runs)s,
                actual_away_runs  = %(actual_away_runs)s,
                delta_er          = %(delta_er)s,
                game_state        = 'Final',
                resolved_at       = %(resolved_at)s
            WHERE game_pk = %(game_pk)s
              AND actual_home_runs IS NULL
        """
        rows = [
            {
                "game_pk": r.game_pk,
                "actual_home_runs": r.actual_home_runs,
                "actual_away_runs": r.actual_away_runs,
                "delta_er": r.delta_er,
                "resolved_at": r.resolved_at,
            }
            for r in results
            if r.error_message is None
        ]

        if not rows:
            logger.warning("save_results_to_db_no_valid_rows")
            return 0

        updated = 0
        try:
            conn = psycopg2.connect(self._dsn)
            cur = conn.cursor()
            for row in rows:
                cur.execute(update_sql, row)
                updated += cur.rowcount
            conn.commit()
            conn.close()
        except psycopg2.Error as exc:
            logger.error("db_save_results_failed", error=str(exc))
            raise

        logger.info("db_results_saved", rows_updated=updated)
        return updated

    # ------------------------------------------------------------------
    # 7. Notificación Slack
    # ------------------------------------------------------------------

    def send_slack_notification(
        self,
        metrics: DayMetrics,
        results: list[GameResult],
        report_path: Path,
    ) -> None:
        """Envía un resumen de métricas al canal de Slack configurado.

        Utiliza Incoming Webhook (Block Kit). Fallo de Slack no es crítico —
        se registra como warning pero no lanza excepción.
        """
        if not self._slack_webhook:
            logger.info("slack_notification_skipped_no_webhook")
            return

        bias_dir = ":arrow_down: subestimó" if not metrics.model_overpredicted else ":arrow_up: sobreestimó"
        status_emoji = ":white_check_mark:" if metrics.mae < 1.0 else ":warning:" if metrics.mae < 2.0 else ":rotating_light:"

        # Líneas de partidos (máx 10 para no superar límite de Slack)
        game_lines = []
        for r in sorted(results, key=lambda x: x.abs_error, reverse=True)[:10]:
            if r.error_message:
                game_lines.append(f"• {r.away_team_name} @ {r.home_team_name} — ⚠ {r.error_message[:40]}")
            else:
                sign = "+" if r.delta_er >= 0 else ""
                game_lines.append(
                    f"• {r.away_team_name} @ {r.home_team_name} "
                    f"| AI: {r.ai_expected_runs:.1f} C | Real: {r.actual_home_runs} C "
                    f"| Δ: {sign}{r.delta_er:.2f}"
                )

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{status_emoji} MLB AI — Informe Post-Partido {metrics.game_date}",
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Partidos resueltos:*\n{metrics.games_resolved}/{metrics.total_games}"},
                    {"type": "mrkdwn", "text": f"*Errores:*\n{metrics.games_with_error}"},
                    {"type": "mrkdwn", "text": f"*MAE E[R]:*\n`{metrics.mae:.4f}` carreras"},
                    {"type": "mrkdwn", "text": f"*RMSE E[R]:*\n`{metrics.rmse:.4f}` carreras"},
                    {"type": "mrkdwn", "text": f"*Sesgo medio:*\n`{metrics.mean_delta_er:+.4f}` ({bias_dir})"},
                    {"type": "mrkdwn", "text": f"*Modelo:*\n`{metrics.model_version}`"},
                ],
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Detalle por partido (ordenado por error):*\n" + "\n".join(game_lines),
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"PDF guardado en `{report_path.name}` · "
                                f"Generado {datetime.now(_ET_TZ).strftime('%H:%M ET')}",
                    }
                ],
            },
        ]

        try:
            resp = requests.post(
                self._slack_webhook,
                json={"blocks": blocks},
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("slack_notification_sent", game_date=metrics.game_date)
        except Exception as exc:
            logger.warning("slack_notification_failed", error=str(exc))
