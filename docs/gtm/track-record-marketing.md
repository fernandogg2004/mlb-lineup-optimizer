# Track Record como Motor de Marketing (Roadmap 4.4)

> "Para un producto de pronóstico, el historial verificable convence más que cualquier demo."
> Fecha: 2026-05-27

---

## Principio central

En el mercado de pronósticos, la credibilidad no se declara, se demuestra.
El pitch estándar ("nuestro modelo tiene X% de precisión") no convence a nadie que haya visto muchas promesas incumplidas.

**Lo que convence:** un track record público donde cualquiera puede comprobar que:
1. Las predicciones se publicaron **antes** del partido (timestamp verificable).
2. Las métricas no se han retocado **a posteriori**.
3. El modelo dice "esto es un coinflip" cuando lo es, y acierta en la frecuencia declarada (calibración).

---

## Cómo usar el track record en cada canal

### Pitch a inversor
- **No empieces con el dashboard.** Empieza con el reliability diagram y el Log-Loss out-of-sample.
- Pregunta retórica: "¿cuántos modelos de pronóstico conoces con track record auditado, no retocado, disponible públicamente?"
- El hecho de que sea verificable públicamente es en sí mismo una señal de calidad.

### Landing page / marketing digital
- Hero: reliability diagram + ECE number como imagen de cabecera.
- CTA: "Ver todas las predicciones históricas (sin filtros, sin cherry-picking)".
- Testimonios: bettors que muestran su ROI usando el modelo.

### Comunidades de Discord / Telegram
- Publicar diariamente las predicciones del día en abierto (con timestamp) y al día siguiente el resultado.
- Los miembros verifican públicamente → construye credibilidad orgánica.
- Nunca borrar predicciones fallidas — el track record completo es el activo.

### Material de venta para sportsbooks (B2B)
- Presentar: Log-Loss vs closing line benchmark + AUC.
- Comparar contra su proveedor actual de datos.
- Ofrecer prueba de 30 días con sus propios datos históricos para validación independiente.

---

## Qué medir y publicar

| Métrica | Por qué importa |
|---------|----------------|
| Log-Loss (rolling 100) | Medida estándar de calibración de probabilidad |
| Brier Score | Complemento al log-loss, interpretable como "error cuadrático" |
| ECE (Expected Calibration Error) | "Cuando digo 70%, ¿pasa el 70%?" — argumento central de venta |
| AUC ROC | Ranking power — ¿distingue partidos ganados de perdidos? |
| CLV (cuando implementado) | Vs closing line — el KPI de un bettor profesional |

---

## Garantías de integridad del track record

Para que el track record sea creíble, debe ser demostrativamente inmutable:
1. **Timestamp en cada predicción** en el momento de generación (antes del partido).
2. **Predicciones en almacenamiento append-only** (results/ folder, git history como prueba).
3. **Backtest.py reproducible:** cualquiera puede clonar el repo y regenerar exactamente las mismas métricas.
4. **Metodología documentada** (backtest.py docstring + docs/backtest-methodology.md).

---

## Cronograma de marketing

| Mes | Acción |
|-----|--------|
| M+0 | Publicar track record público (página /track-record) |
| M+1 | Comunidad Discord con predicciones diarias + track record en vivo |
| M+2 | Primera campaña en comunidades de picks LatAm |
| M+3 | Pitch a primer sportsbook LatAm con 90 días de track record verificable |
| M+6 | Case study con bettor que demuestre ROI positivo usando el modelo |
