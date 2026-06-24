"""backtest_lineups.py — Valor real del optimizador de orden de bateo (Fase 3).

El diagnóstico de Fase 0/2 mostró que predecir el GANADOR a nivel juego tiene un
techo intrínseco bajo (MLB pregame ≈ moneda). Pero el PROPÓSITO real del sistema
no es ese: es ordenar la alineación para maximizar carreras. Ahí el PA model
calibrado + el motor de carreras calibrado (sesgo +0.008) SÍ aportan, porque el
objetivo es el E[R] RELATIVO entre órdenes, no acertar un binario ruidoso.

Este backtest cuantifica ese valor de forma contrafactual, reusando los lineups
ya calculados (``results/<fecha>/*.json`` guardan el ``batting_order`` con el
vector de probabilidades PA de cada bateador — no hace falta re-ejecutar el
modelo). Para cada equipo-juego compara, medido con el MISMO motor de alta
precisión, el E[R] de:

  - orden ÓPTIMO (GeneticLineupOptimizer real, el de producción),
  - orden USADO (el guardado),
  - órdenes ALEATORIOS (distribución de referencia).

Y responde: ¿cuánto sube el óptimo sobre el usado? ¿el usado ya está por encima
del azar? ¿el uplift supera el ruido Monte Carlo?

Uso:
    python scripts/backtest_lineups.py
    python scripts/backtest_lineups.py --max-games 40 --n-random 30
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.constants import LEAGUE_AVG_LINEUP  # noqa: E402
from src.simulation.simulation_engine import MonteCarloConfig, MonteCarloEngine  # noqa: E402
from src.optimizer.lineup_optimizer import GAConfig, GeneticLineupOptimizer, PlayerStats  # noqa: E402

RESULTS_DIR = ROOT / "results"
DEFAULT_OUT = ROOT / "reports" / "diagnostics" / "lineup_backtest.json"

_PROB_KEYS = ["prob_out", "prob_k", "prob_bb", "prob_1b", "prob_2b", "prob_3b", "prob_hr", "prob_dp"]


def _parse_lineup(order: list[dict]) -> tuple[list[PlayerStats], np.ndarray] | None:
    """Reconstruye PlayerStats + matriz (9,8) desde un ``batting_order`` guardado.

    Args:
        order: Lista de dicts con id, name, prob_* y stats por bateador.

    Returns:
        ``(players, probs)`` o ``None`` si el orden no tiene exactamente 9
        bateadores con vector de probabilidad completo.
    """
    if len(order) != 9:
        return None
    players: list[PlayerStats] = []
    rows: list[list[float]] = []
    for p in order:
        if not all(k in p for k in _PROB_KEYS):
            return None
        vec = [float(p[k]) for k in _PROB_KEYS]
        # iso aproximada desde el vector (bases extra por PA); batter_stand no se
        # guarda — sólo afecta al SEEDING del GA, no a la evaluación de E[R].
        iso = float(p["prob_2b"] + 2 * p["prob_3b"] + 3 * p["prob_hr"])
        players.append(PlayerStats(
            player_id=int(p.get("id", 0)),
            player_name=str(p.get("name", "")),
            obp=float(p.get("obp_est", 0.320)),
            woba=float(p.get("woba_stab", 0.318)),
            iso=iso,
            batter_stand=str(p.get("stand", "R")),
            prob_vector=np.asarray(vec, dtype=np.float32),
        ))
        rows.append(vec)
    return players, np.asarray(rows, dtype=np.float32)


def _er(engine: MonteCarloEngine, probs: np.ndarray) -> float:
    """E[R] de un orden vía la ruta rápida (E[R] no depende del rival)."""
    return engine.run_fast(probs.astype(np.float32), LEAGUE_AVG_LINEUP.astype(np.float32))


def run_lineup_backtest(max_games: int, n_random: int, out_path: Path) -> dict:
    """Ejecuta el backtest contrafactual de órdenes de bateo.

    Args:
        max_games: Máximo de equipo-juegos a evaluar.
        n_random: Nº de órdenes aleatorios por juego para la distribución de azar.
        out_path: Ruta del JSON de salida.

    Returns:
        Dict con los resultados por juego y los agregados.
    """
    files = sorted(glob.glob(str(RESULTS_DIR / "2*" / "*.json")))
    # Motor de evaluación de ALTA precisión (idéntico para todos los órdenes).
    eval_engine = MonteCarloEngine(MonteCarloConfig(
        use_ray=False, fast_mode_n_sims=60_000, use_extra_innings=False,
    ))
    # Motor del optimizador (config de producción reducida para latencia).
    opt_engine = MonteCarloEngine(MonteCarloConfig(
        use_ray=False, n_simulations=20_000, fast_mode_n_sims=4_000, use_extra_innings=False,
    ))
    ga_cfg = GAConfig(
        n_generations=30, population_size=60, n_sabermetric_seeds=15,
        refinement_top_k=5, objective="expected_runs",
    )
    rng = np.random.default_rng(42)

    records: list[dict] = []
    for fp in files:
        if len(records) >= max_games:
            break
        try:
            d = json.loads(Path(fp).read_text(encoding="utf-8"))
        except Exception:
            continue
        order = d.get("batting_order")
        if not order:
            continue
        parsed = _parse_lineup(order)
        if parsed is None:
            continue
        players, probs_saved = parsed

        # E[R] del orden USADO (alta precisión)
        er_saved = _er(eval_engine, probs_saved)

        # Orden ÓPTIMO con el GA real de producción
        try:
            optimizer = GeneticLineupOptimizer(
                players, probs_saved, opt_engine,
                LEAGUE_AVG_LINEUP.astype(np.float32), config=ga_cfg,
            )
            res = optimizer.run(pitcher_hand=str(d.get("opp_pitcher_throws", "R") or "R"))
            best_idx = res.best_lineup_indices
            probs_opt = probs_saved[best_idx]
            er_opt = _er(eval_engine, probs_opt)   # re-evaluado con el motor común
            is_sig = bool(res.is_significant_vs_second)
        except Exception as e:
            print(f"  [WARN] optimizer falló en {Path(fp).name}: {e}")
            continue

        # Distribución ALEATORIA (mismo motor de evaluación)
        rand_ers = []
        for _ in range(n_random):
            perm = rng.permutation(9)
            rand_ers.append(_er(eval_engine, probs_saved[perm]))
        rand_ers = np.asarray(rand_ers)
        saved_pct = float((rand_ers < er_saved).mean())   # percentil del usado vs azar

        records.append({
            "file": Path(fp).name,
            "game_pk": d.get("game_pk"),
            "team": d.get("team_abbr"),
            "er_saved": round(er_saved, 4),
            "er_optimal": round(er_opt, 4),
            "er_random_mean": round(float(rand_ers.mean()), 4),
            "er_random_best": round(float(rand_ers.max()), 4),
            "uplift_opt_vs_saved": round(er_opt - er_saved, 4),
            "uplift_opt_vs_random_mean": round(er_opt - float(rand_ers.mean()), 4),
            "saved_vs_random_mean": round(er_saved - float(rand_ers.mean()), 4),
            "saved_percentile_vs_random": round(saved_pct, 3),
            "optimizer_significant_vs_2nd": is_sig,
        })
        print(f"  {len(records):>3}. {d.get('team_abbr','?'):<4} "
              f"saved={er_saved:.3f} opt={er_opt:.3f} rand={rand_ers.mean():.3f}  "
              f"uplift(opt-saved)={er_opt-er_saved:+.3f}", flush=True)

    if not records:
        print("  [ERROR] No se evaluó ningún lineup (¿faltan batting_order en results/?).")
        return {"error": "sin lineups evaluables", "n": 0}

    # ── Agregados con IC (across games) ──────────────────────────────────────
    def _ci(vals: list[float]) -> dict:
        a = np.asarray(vals)
        mean = float(a.mean())
        se = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0
        return {"mean": round(mean, 4), "se": round(se, 4),
                "ci90_low": round(mean - 1.645 * se, 4), "ci90_high": round(mean + 1.645 * se, 4)}

    up_os = _ci([r["uplift_opt_vs_saved"] for r in records])
    up_or = _ci([r["uplift_opt_vs_random_mean"] for r in records])
    sv_r = _ci([r["saved_vs_random_mean"] for r in records])
    saved_pcts = np.asarray([r["saved_percentile_vs_random"] for r in records])

    summary = {
        "n_games": len(records),
        "uplift_optimal_vs_saved_runs": up_os,
        "uplift_optimal_vs_random_runs": up_or,
        "saved_vs_random_runs": sv_r,
        "saved_order_avg_percentile_vs_random": round(float(saved_pcts.mean()), 3),
        "frac_optimizer_significant_vs_2nd": round(
            float(np.mean([r["optimizer_significant_vs_2nd"] for r in records])), 3),
        "approx_runs_per_season_if_persisted": round(up_os["mean"] * 162, 1),
    }

    print("\n=== RESUMEN ===", flush=True)
    print(f"  n={summary['n_games']}", flush=True)
    print(f"  Uplift óptimo vs usado:  {up_os['mean']:+.3f} runs/juego  "
          f"IC90 [{up_os['ci90_low']:+.3f}, {up_os['ci90_high']:+.3f}]", flush=True)
    print(f"  Uplift óptimo vs azar:   {up_or['mean']:+.3f} runs/juego  "
          f"IC90 [{up_or['ci90_low']:+.3f}, {up_or['ci90_high']:+.3f}]", flush=True)
    print(f"  Usado vs azar:           {sv_r['mean']:+.3f} runs/juego  "
          f"(percentil medio del usado: {summary['saved_order_avg_percentile_vs_random']})", flush=True)
    print(f"  ~runs/temporada (162) si se aplicara el óptimo: "
          f"{summary['approx_runs_per_season_if_persisted']:+}", flush=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"max_games": max_games, "n_random": n_random,
                   "eval_fast_sims": 60_000, "ga_generations": 30},
        "summary": summary,
        "games": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado -> {out_path}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest del valor del optimizador de lineup (Fase 3)")
    parser.add_argument("--max-games", type=int, default=40)
    parser.add_argument("--n-random", type=int, default=30)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()
    run_lineup_backtest(args.max_games, args.n_random, Path(args.out))
