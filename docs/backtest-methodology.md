# Metodología de Backtest — MLB War Room

> Este documento explica cómo se calculan las métricas de rendimiento del modelo y
> cómo reproducirlas de forma independiente.
> Fecha: 2026-05-27

---

## Principio de separación temporal estricta

**El test set NUNCA ha visto el modelo durante el entrenamiento.**

- El modelo (`pa-model-v3`) fue congelado después del entrenamiento en datos hasta [fecha de corte].
- Las predicciones del test set se generaron corriendo el modelo congelado sobre partidos **posteriores** al corte.
- Ninguna predicción en el track record se retocó a posteriori.

---

## Cómo reproducir el backtest

```bash
# Desde la raíz del proyecto:
python backtest.py

# Con rango de fechas específico:
python backtest.py --from 2026-04-01 --date 2026-05-27

# Salida a fichero específico:
python backtest.py --out reports/backtest/backtest_results_v2.json
```

El script genera `reports/backtest/backtest_results.json` con:
- Métricas agregadas (Log-Loss, Brier Score, AUC ROC, ECE)
- Bins de calibración (10 intervalos de probabilidad 0-100%)
- Lista de todas las predicciones con timestamps y resultados

---

## Métricas calculadas

### Log-Loss (pérdida logarítmica)
```
LL = -1/N × Σ [y_i × log(p_i) + (1-y_i) × log(1-p_i)]
```
- `y_i = 1` si ganó el equipo local, `0` si perdió
- `p_i` = probabilidad de victoria del equipo local predicha por el modelo
- **Baseline aleatorio:** LL = ln(2) ≈ 0.693
- **Baseline Elo-style MLB:** ≈ 0.431
- **Umbral de regresión:** > 0.450 → alerta en tests

### Brier Score
```
BS = 1/N × Σ (p_i - y_i)²
```
- Interpretable como error cuadrático medio de las probabilidades
- **Baseline aleatorio (siempre 0.5):** BS = 0.25
- **Umbral de regresión:** > 0.250

### AUC ROC
- Mide la capacidad discriminante (ranking): ¿separa el modelo ganadores de perdedores?
- **Baseline aleatorio:** AUC = 0.50
- **Umbral de regresión:** < 0.52

### ECE (Expected Calibration Error)
```
ECE = Σ_b (|B_b| / N) × |acc(B_b) - conf(B_b)|
```
- `B_b` = bin de probabilidad b, `acc` = frecuencia observada, `conf` = probabilidad media predicha
- **Interpretación:** cuando el modelo dice 70%, ¿pasa el 70% de las veces?
- **ECE = 0** = calibración perfecta
- **Umbral alerta:** ECE > 0.05 → revisar calibración
- **Umbral fallo:** ECE > 0.10 → degradación severa

---

## Integridad del track record

1. Cada predicción se genera y guarda en `results/{TEAM}_{YYYY-MM-DD}.json` **antes** del partido.
2. El timestamp del fichero (st_mtime) es prueba de anterioridad.
3. El repositorio Git mantiene el historial de commits — cualquier modificación posterior queda registrada.
4. Los ficheros de resultados son append-only: nunca se sobreescriben predicciones pasadas.

---

## Fuentes de datos del test

| Fuente | Uso |
|--------|-----|
| `results/*.json` | Predicciones generadas por el modelo (win_probability, E[R]) |
| `reports/comparison/comparison_*.json` | Resultados reales de cada partido |
| MLB Stats API | Fallback para obtener resultados si no hay comparison file |

---

## Limitaciones conocidas

1. **n < 162:** el modelo lleva activo desde [fecha inicio de temporada]. El backtest crece conforme avanzan los partidos.
2. **Platoon y fatiga no incluidos:** el modelo actual no ajusta por matchup platoon ni fatiga. El Roadmap 3.1 añade fatiga solo si mejora el Log-Loss out-of-sample.
3. **Un partido = una predicción:** se deduplicó a una predicción por partido (lado home preferido). No hay múltiples predicciones por partido.
