# Fase 3 — Valor real del optimizador de lineup

Fecha: 2026-06-24
Artefactos: `reports/diagnostics/lineup_backtest.json`
Código: `scripts/backtest_lineups.py` (nuevo), `predict_tonight.py` (guard de completitud)

## Resumen

El nivel-juego es una moneda (Fase 0/2), pero el PROPÓSITO del sistema —ordenar la alineación—
**sí entrega valor medible y estadísticamente significativo**. El backtest contrafactual reusa los
lineups ya guardados (`results/<fecha>/*.json` traen el `batting_order` con el vector PA de cada
bateador, así que no se re-ejecuta el modelo) y compara, con el MISMO motor de alta precisión, el
E[R] del orden óptimo (GeneticLineupOptimizer real) vs el usado vs órdenes aleatorios.

## Resultados (n=40 equipo-juegos)

| Comparación | Uplift (runs/juego) | IC90 | Significativo |
|---|---|---|---|
| Óptimo vs orden usado | **+0.052** | [+0.033, +0.072] | sí (excluye 0) |
| Óptimo vs aleatorio | +0.103 | [+0.082, +0.124] | sí |
| Usado vs aleatorio | +0.050 | — | usado en percentil 81 |

- **~+8.5 runs/temporada** (162 juegos) si se aplicara el óptimo sobre el orden usado (~1 victoria).
- Concuerda con la literatura sabermétrica (optimizar el orden vale ~5–15 runs/temporada).
- El orden "usado" ya es razonable (percentil 81 vs azar); el optimizador ~duplica el margen sobre
  el azar. Los mayores uplifts aparecen en juegos de alta ofensiva (SEA +0.38, NYY +0.20, CHC +0.21),
  donde el orden tiene más palanca — comportamiento esperado.

## Interpretación

A diferencia del win-prob a nivel juego (techo intrínseco ≈ moneda), el E[R] RELATIVO entre órdenes
es justo donde el PA model calibrado (ECE 0.002) + el motor de carreras calibrado (sesgo +0.008)
aportan. **El sistema cumple su propósito principal con valor estadísticamente probado.**

Caveat honesto: la validación es contrafactual (E[R] simulado), no resultado real por-juego — el
efecto del orden (~0.05 runs/juego) es demasiado pequeño para verlo en una sola realización ruidosa.
Pero el motor de E[R] está calibrado de forma independiente (Fase 2), lo que respalda el contrafactual.

## Guard de completitud en serving (higiene de datos)

`predict_tonight.py` ahora etiqueta cada predicción guardada con `sim_status`:
- `"two_sided"`: win_probability de una simulación consistente de dos lados (P(home)+P(away)≈1).
- `"vs_league_avg"`: sólo un lineup disponible; win_probability vs rival promedio (NO comparable).
- `"no_sim"`: sin simulación.

Esto evita que vuelva a colarse el ~52% de predicciones a medio simular que contaminó el backtest
histórico (y que producía los síntomas falsos de accuracy 24.9% / win-prob descentrado).
`backtest.py` ya exige predicción válida de dos lados.
