# Fase 2 — Diagnóstico del nivel juego (simulador + win-prob)

Fecha: 2026-06-24
Artefactos: `reports/backtest/backtest_clean.json`, `reports/diagnostics/ceiling_test.json`
Cambios de código: `backtest.py` (filtro de completitud + IC + sim-vs-real + forma win-prob)

## Resumen

Se planificaban tres arreglos de Fase 2 (inflación de carreras, centrado del win-prob, blend).
La medición controlada **descartó los dos primeros como no-bugs** y descubrió un **problema de
higiene de datos** que contaminaba toda la evaluación. La única palanca real de ventaja a nivel
juego sigue siendo el **blend con fuerza de equipo/Elo**, con expectativa de ganancia pequeña
(MLB pregame es intrínsecamente casi cara-o-cruz).

## Lo que se verificó

### 1. Inflación de carreras del simulador → NO existe
Test controlado (lineup promedio-de-liga, park neutral, 100k sims):
- prob advances + extra innings (PROD): E[R] 4.520
- **prob advances, 9 innings: E[R] 4.381 vs 4.373 reales → sesgo +0.008 (calibrado)**
- determ advances: 3.759 (por eso existen las tablas probabilísticas)

El "+0.47" del backtest inicial era artefacto: comparaba E[R] contra carreras del *local*, que
rindió por debajo de la media en esa muestra. El motor de 9 innings está bien calibrado. El path
de serving (`_run_game_simulation`) ya usa `use_extra_innings=False`, así que no hay offset de
extras en producción. **No se toca el motor de carreras.**

### 2. Centrado del win-prob → NO existe (era contaminación)
En juegos con ambos lineups simulados:
- P(home)+P(away) = 1.000, corr(E[R] dif, winprob−0.5) = 0.996 → internamente consistente.
- home win-prob: media 0.502, frac>0.5 = 0.510 → **bien centrado**.

### 3. Bug REAL encontrado: higiene de datos del backtest
El 52% de los juegos (115 de 221) tenían un lado **sin simular** (`win_probability` ausente) y el
dedup lo rellenaba con 0.5 por defecto. Esas predicciones degeneradas producían los síntomas
falsos: accuracy direccional 24.9%, win-prob "descentrado" (24% > 0.5), ECE 0.139.
**Arreglo**: `backtest.py` ahora exige predicción válida de dos lados (P(home)+P(away)≈1) y
excluye e informa las incompletas.

## Estado real del sistema (104 juegos válidos, IC90 bootstrap)
- Log-Loss 0.6986, IC90 [0.665, 0.732] → el 50/50 (0.6931) cae DENTRO.
- Delta vs 50/50: −0.0055, IC90 [−0.039, +0.028], P(modelo mejor) = 0.41.
- AUC 0.538, IC90 [0.446, 0.631] → cruza 0.5.
- Win-prob centrado (51.0%), accuracy 51.9%, beats Elo trivialmente (0.6986 vs 0.6994).
- Sim-vs-real: corr(E[R], carreras reales) ≈ 0.016 → la ofensiva no tiene señal a nivel juego.

**Conclusión**: el sistema es estadísticamente una moneda a nivel juego. No por bugs (no los hay),
sino porque (a) MLB pregame es casi aleatorio y (b) la discriminación del PA model es modesta
(prueba de techo). Elo mismo apenas supera al 50/50 aquí.

## Trabajo restante de Fase 2
1. **(Hecho) Higiene de datos** en el backtest.
2. **Blend con prior de fuerza de equipo/Elo** — única vía con potencial de ventaja a nivel juego.
   Expectativa honesta: mejora pequeña (log-loss ~0.685–0.69). Valor adicional: no quedar por
   debajo del 50/50 y aportar el poco edge agregable. Idealmente, además, asegurar en serving que
   ningún juego se guarde con un lado sin simular (espejo del filtro del backtest, para producción).

## Recomendación
Dado el techo intrínseco a nivel juego, repartir esfuerzo: implementar el blend Elo (barato, cierra
Fase 2) y mover el foco principal a **Fase 3 (valor del lineup)**, donde el PA model calibrado + el
motor de carreras calibrado SÍ aportan (E[R] relativo entre órdenes), que es el propósito real del
sistema.
