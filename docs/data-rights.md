# Derechos de Datos — MLB War Room

> **Propósito:** respuesta clara y por escrito a "¿de dónde sacas los datos y puedes usarlos comercialmente?".
> Última actualización: 2026-05-27

---

## Fuentes de datos activas

### 1. MLB Stats API (statsapi.mlb.com)
- **URL base:** `https://statsapi.mlb.com/api/v1`
- **Tipo:** API pública no autenticada
- **Datos usados:**
  - Partidos del día (`/schedule`)
  - Lineups y alineaciones confirmadas (`/game/{gamePk}/linescore`, `/boxscore`)
  - Estadísticas de jugadores (AVG, OBP, OPS, ISO, wOBA estimado)
  - Datos de pitcheo (ERA, mano)
- **Términos:** La MLB Stats API es de acceso libre sin autenticación para uso informativo. Para uso **comercial** (distribución de datos, integración en producto de pago), los Términos de Servicio de MLB Advanced Media requieren acuerdo de licencia.
  - Referencia: [MLB Developer Portal](https://developer.sportradar.com/) y [MLB.com Terms of Use](https://www.mlb.com/official-information/terms-of-use)
- **Acción requerida antes de comercializar:** contactar a MLB Advanced Media (MLBAM) para licencia de datos comercial.

### 2. Baseball Savant / Statcast
- **URL base:** `https://baseballsavant.mlb.com/`
- **Tipo:** Portal público de MLB, datos de Statcast
- **Datos usados actualmente:** ninguno directo en tiempo real. Los datos wOBA, ISO, OBP que se muestran provienen de la MLB Stats API, no de scraping de Savant.
- **Uso futuro (modelo de fatiga):** si se incorporan datos de Statcast (pitch velocity, exit velocity, sprint speed), aplican los mismos términos de MLBAM.
- **Acción requerida:** si se incorpora Statcast, revisar si el volumen y uso caen dentro del uso no comercial/investigación o requieren licencia.

### 3. Datos de mercado de apuestas (no integrado actualmente)
- Si se implementa la capa económica (Roadmap 3.2 — CLV / EV vs línea de cierre), los feeds de odds proceden de fuentes como The Odds API, Pinnacle API, etc.
- Cada proveedor tiene sus propios términos. The Odds API permite uso comercial con plan de pago.

---

## Conclusión ejecutiva

| Fuente                     | Uso actual   | Uso comercial permitido hoy |
|----------------------------|--------------|-----------------------------|
| MLB Stats API              | Sí           | Requiere licencia MLBAM     |
| Statcast / Baseball Savant | No (todavía) | Requiere licencia MLBAM     |
| Datos de odds              | No           | Depende del proveedor (The Odds API: sí con plan) |

**Riesgo principal:** usar MLB Stats API en un producto de pago sin acuerdo de licencia. Antes de cerrar cualquier cliente de pago, se debe:
1. Contactar a MLB Advanced Media para negociar licencia de datos.
2. Evaluar alternativas de datos con licencia comercial clara (Sports Radar, Stats Perform, etc.).

**Dato positivo:** todos los modelos predictivos, la lógica de optimización y las métricas de calibración son propiedad del equipo desarrollador y no están sujetos a restricciones externas.

---

## Referencias legales

- [MLB Terms of Use](https://www.mlb.com/official-information/terms-of-use)
- [MLB Advanced Media Licensing](https://www.mlbam.com/)
- [Sports Radar MLB Data Feed](https://sportradar.com/sports-data/baseball/)
- [The Odds API Terms](https://the-odds-api.com/terms-of-use/)
