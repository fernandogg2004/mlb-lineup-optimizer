# Fase 1 — Discriminación del PA model: techo de datos

Fecha: 2026-06-24
Artefactos: `reports/diagnostics/ceiling_test.json`, `reports/diagnostics/feature_screen.json`
Código: `scripts/feature_screen.py` (nuevo)

## Objetivo
Subir la discriminación del PA model (AUC OOS 0.571) SIN romper la calibración, añadiendo
información NUEVA. Disciplina: screening barato de AUC OOS antes de integrar en el contrato anti-skew.

## Realización técnica que acotó el enfoque
Añadir interacciones de features ya presentes (same-hand, platoon×park, tendencias, EWMAs) NO sube
el AUC de un GBM: LightGBM ya captura interacciones y diferencias por árboles (la prueba de techo V0
ya deja interactuar las 51). Sólo INFORMACIÓN NUEVA puede mover la aguja.

## Disponibilidad de datos (auditada)
- `data/raw`: pitch-level pero sólo 2021-2024, sin velo/spin, sin `game_pk` → no servible en 2026.
- Weather: sin datos en disco; el `weather_lambda` es realtime → backfill histórico = ingesta nueva.
- Silver (todas las temporadas): xwoba/launch por PA, ya explotados como medias móviles en las 51.

## Screening (mismo split que la prueba de techo: train ≤2024, eval 2026 OOS)
Candidatas con info nueva, derivables de todas las temporadas y servibles en producción:

| Feature añadida a V0 | wAUC | Δ vs V0 |
|---|---|---|
| V0 (control, 51 features) | 0.5701 | — |
| + pitcher_xwoba_allowed_30d | 0.5703 | +0.0002 |
| + pitcher_hard_hit_allowed_30d | 0.5708 | +0.0007 |
| + batter_days_rest | 0.5704 | +0.0003 |
| + pitcher_days_rest | 0.5712 | +0.0011 |
| + TODAS | 0.5707 | +0.0006 |

**Ninguna alcanza el umbral de +0.003.** La calidad de contacto permitida y la fatiga ya están
implícitas en las features de pitcher/bateador existentes.

## Conclusión
**El PA model está en su techo de discriminación dado los datos disponibles.** Combinado con la
prueba de techo (identidad no ayuda; el techo lo cierra el contacto del propio PA, casi estocástico),
la evidencia es consistente: para más discriminación se necesita una **NUEVA fuente de datos**, no
mejor ingeniería sobre la existente.

## Opciones reales para más discriminación (todas requieren ingesta nueva)
1. **Statcast "stuff" completo** (release_speed, spin_rate, movimiento, arsenal por tipo de pitch)
   vía pybaseball para 2015-2026 → features de arsenal/stuff del pitcher y de vulnerabilidad del
   bateador por tipo de pitch. Mayor potencial, pero esfuerzo de data-engineering (GBs, pipeline,
   re-screen) y ganancia aún acotada por la estocasticidad del contacto.
2. **Weather backfill** (temp, viento, humedad por estadio×fecha) → efecto en HR/contacto.
3. **Aceptar el techo**: el modelo es tan bueno como permiten los datos; el valor del sistema ya
   está capturado en Fase 3 (optimización de lineup, +8.5 runs/temporada).

Recomendación: no integrar nada de lo screened (cero ganancia). Decidir si se financia la ingesta
nueva (opción 1 es la de mayor potencial) o se acepta el techo.
