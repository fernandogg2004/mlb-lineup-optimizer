# Checkpoint 0 — Diagnóstico del pipeline (Fase 0)

Fecha: 2026-06-24
Artefactos: `reports/diagnostics/pipeline_diagnosis.json`, `reports/backtest/backtest_full_window.json`
Scripts: `scripts/diagnose_pipeline.py` (PA-OOS + dispersión win-prob), `backtest.py` (nivel juego + IC bootstrap)

## TL;DR

El modelo PA está **perfectamente calibrado pero no discrimina**, y eso se propaga: a nivel
juego el sistema es **estadísticamente indistinguible de lanzar una moneda**. El "pierde contra
Elo (0.431)" del snapshot de 45 juegos era **ruido de muestra pequeña** — con 217 juegos el Elo
da 0.6973 y el modelo 0.6958 (ambos ≈ ln 2). Hay además dos defectos del simulador encima.

## Evidencia

### A. PA model OOS (temporada 2026, 82,574 PA nunca vistos — train ≤2024, val 2025)
- **Calibración impecable**: ECE **0.0021** (target ≤0.035), sesgo E[R/PA] **−0.0006**.
- **Discriminación pobre**: AUC ponderado **0.571**. Por clase: HR 0.607, K 0.600, BB 0.590,
  OUT 0.564, 1B 0.543, 2B 0.525. El modelo predice ~tasas de liga para casi todos.

### B. Nivel juego (217 juegos, IC90 bootstrap)
- **Log-Loss 0.6958, IC90 [0.679, 0.713]** → el 50/50 (0.6932) cae DENTRO del intervalo.
- **Delta vs 50/50: −0.0026, IC90 [−0.019, +0.015], P(modelo mejor) = 0.41** → sin ventaja medible.
- **AUC 0.519, IC90 [0.457, 0.584]** → el intervalo cruza 0.5: sin discriminación.

### C. Defectos del simulador (independientes del PA model)
- **Inflación de carreras**: E[R] simulado 4.65 vs 4.18 reales → **+0.47 runs/juego**.
- **Correlación E[R]↔carreras reales ≈ −0.10** → la estimación de ofensiva no tiene señal útil.
- **Win-prob mal centrado**: media 0.50 pero sólo **24%** de las probabilidades superan 0.5
  (mediana ≈ 0.50, distribución sesgada) → cualquier decisión con umbral 0.5 se rompe
  (accuracy direccional 24.9%).

## Diagnóstico de la causa raíz

1. **Raíz (Fase 1): falta de discriminación del PA model.** Si no se puede saber qué PA será un
   HR/1B, no se puede saber qué equipo anota más → el win-prob colapsa hacia 0.5.
2. **Insight metodológico clave**: el modelo se entrena y se selecciona (gate) **sólo por
   calibración** (ECE + sesgo E[R]); la **discriminación nunca fue objetivo ni gate**. Se optimizó
   justo la propiedad que tiene y se ignoró la que da valor a nivel juego. Las features son
   promedios fuertemente encogidos (James-Stein + regularización) que por diseño regresan a la
   media y matan la discriminación.
3. **Secundario (Fase 2): calibración del simulador** — inflación de +0.47 runs/juego y win-prob
   descentrado, ambos arreglables aguas abajo e independientes del PA model.

## Implicación para el roadmap

- **Fase 1 (PA model) sube de prioridad** con un giro: el objetivo deja de ser "mantener ECE" (ya
  es óptima) y pasa a ser **subir discriminación** (AUC/log-loss por clase) SIN romper la
  calibración. Palancas: relajar shrinkage, features de matchup más nítidas (weather, contacto,
  vulnerabilidad por tipo de pitch), interacciones platoon×park.
- **Fase 2 (simulador)** mantiene dos arreglos concretos y medibles ya cuantificados: quitar la
  inflación de +0.47 runs y re-centrar/calibrar el win-prob a nivel juego.
- **Cautela honesta**: parte del techo de discriminación a nivel PA es intrínseco al béisbol
  (alta varianza). El criterio de éxito realista no es "ganar mucho" sino **batir de forma
  estadísticamente significativa al 50/50 y al Elo** en el backtest amplio con IC.

## Prueba de techo (Fase 0bis) — `reports/diagnostics/ceiling_test.json`

Split: train ≤2024, early-stop 2025, eval 2026 OOS. Métrica = discriminación (AUC OvR).

| Variante | AUC pond. | HR | K | 1B | best_iter |
|---|---|---|---|---|---|
| V0 actual (control) | 0.570 | 0.605 | 0.598 | 0.541 | 70 |
| V1 + identidad (batter/pitcher_id) | 0.561 | 0.582 | 0.592 | 0.531 | 22 |
| V2 + contacto del PA (xwoba/launch, leaking) | 0.954 | 0.989 | 0.999 | 0.927 | 154 |
| V3 + identidad + contacto | 0.951 | 0.988 | 0.999 | 0.923 | 89 |

Conclusiones:
- **V0 = 0.570 reproduce el modelo desplegado (0.571)** → control válido.
- **La identidad NO aporta (V1 ≤ V0)** → REFUTA "relajar shrinkage / memorizar al jugador".
  Los agregados encogidos ya capturan la parte predecible del skill; la identidad sobreajusta.
- **El techo (V2/V3 ≈ 0.95) está cerrado por la calidad de contacto del PROPIO PA**, que es
  post-swing y casi estocástica. (Caveat: K=0.999 es parte artefacto — launch null en strikeouts;
  pero HR=0.989 y 1B=0.927 son legítimos.)
- Lectura: el espacio 0.57→0.95 es "contacto". La palanca de Fase 1 NO es relajar shrinkage sino
  **forecasting de contacto pre-swing** (matchup vs tipo de pitch, stuff/arsenal, park, weather).
  Ganancia esperada **modesta** (pocos puntos de AUC), no salto a 0.95.

## Replanteo del roadmap tras la prueba de techo

- **Fase 1 (PA)**: objetivo = subir discriminación con features de matchup/contacto pre-swing;
  expectativa de ganancia modesta y acotada. Descartado: relajar shrinkage / identidad.
- **Fase 2 (juego) sube de prioridad relativa**: arreglar inflación del simulador (+0.47 runs),
  re-centrar/calibrar win-prob y **blend con prior de fuerza de equipo/Elo** (agrega miles de PA,
  mucho más predecible que un PA). Es la vía más fiable para ganar al 50/50 a nivel juego.
- **Fase 3 (negocio)**: el optimizador necesita E[R] RELATIVO bien ordenado entre lineups, no
  acertar el ganador — objetivo más alcanzable y donde el PA model calibrado ya aporta.

## Decisión pendiente (este checkpoint)
¿Atacamos primero la discriminación del PA model (Fase 1) o los arreglos de calibración del
simulador (Fase 2, más rápidos y ya cuantificados)? Recomendación: **Fase 1 primero** (es la raíz),
con los arreglos del simulador en paralelo por ser baratos y de señal limpia.
