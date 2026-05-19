"""
api/shadow.py
=============
Shadow Mode — ejecuta el modelo challenger en paralelo al champion sin
afectar las respuestas que recibe el cliente.

Flujo:
  1. Champion responde normalmente al cliente.
  2. En background, ShadowPredictor corre el challenger con los mismos inputs.
  3. Los resultados del challenger se almacenan en la tabla `shadow_predictions`
     de PostgreSQL para comparación offline.
  4. Cuando el challenger supera al champion (ECE < champion ECE), se ejecuta
     promote_model.py para promoverlo a producción.

Activación: configurar la variable de entorno SHADOW_MODEL_PATH antes de arrancar la API.
"""
from __future__ import annotations

import asyncio
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("mlb_api.shadow")

DB_URL = os.getenv("DATABASE_URL", "postgresql://mlb:mlb@postgres:5432/mlb_predictions")


class ShadowPredictor:
    """Wrapper que ejecuta el modelo challenger de forma asíncrona."""

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._predictor  = self._load(model_path)
        log.info("ShadowPredictor cargado desde %s", model_path)

    @staticmethod
    def _load(path: str):
        sys.path.insert(0, str(Path(path).parent.parent))
        import src.models.model_at_bat as _mat
        for _n in dir(_mat):
            setattr(sys.modules["__main__"], _n, getattr(_mat, _n))
        with open(path, "rb") as f:
            payload = pickle.load(f)
        predictor = _mat.AtBatPredictor(config=payload["config"])
        predictor._calibrated_model = payload["calibrated_model"]
        predictor._base_model       = payload["base_model"]
        predictor._feature_names    = payload["feature_names"]
        predictor._is_fitted        = True
        return predictor

    def run_async(self, game: dict, side: str, silver, game_date: str) -> None:
        """Lanza la inferencia challenger en un hilo background (fire-and-forget)."""
        try:
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, self._run_shadow, game, side, silver, game_date)
        except Exception as exc:
            log.warning("Shadow mode dispatch falló: %s", exc)

    def _run_shadow(self, game: dict, side: str, silver: Any, game_date: str) -> None:
        try:
            t0 = time.perf_counter()
            from predict_tonight import _predict_one_side
            result = _predict_one_side(game, side, silver, self._predictor, game_date, verbose=False)
            elapsed = time.perf_counter() - t0
            if result:
                self._store_shadow_result(result, elapsed)
        except Exception as exc:
            log.warning("Shadow inference falló para gamePk=%s side=%s: %s",
                        game.get("gamePk"), side, exc)

    def _store_shadow_result(self, result: dict, latency_s: float) -> None:
        import json
        try:
            import psycopg2
            conn = psycopg2.connect(DB_URL)
            cur  = conn.cursor()
            cur.execute("""
                INSERT INTO shadow_predictions
                    (game_date, game_pk, team_abbr, side, batting_order,
                     expected_runs, model_path, latency_ms, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT DO NOTHING;
            """, (
                result["game_date"],
                result["game_pk"],
                result["team_abbr"],
                result["side"],
                json.dumps(result["batting_order"]),
                result["expected_runs_per_game"],
                self._model_path,
                round(latency_s * 1000, 1),
            ))
            conn.commit()
            conn.close()
        except Exception as exc:
            log.warning("Shadow DB write falló: %s", exc)
