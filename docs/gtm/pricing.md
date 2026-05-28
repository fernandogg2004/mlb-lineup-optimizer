# Pricing y Empaquetado (Roadmap 4.3)

> Modelo de monetización coherente con el segmento beachhead (apuestas/DFS LatAm).
> Fecha: 2026-05-27

---

## Premisa de pricing

Un bettor compra edge. Si el modelo tiene CLV > 0, está justificado cobrar una fracción de ese valor.
La calibración pública es el argumento de venta: nadie paga por un modelo que no puede probar su precisión.

**Fórmula de valor:** un bettor que apuesta €100 por partido, 3 partidos/día, con un CLV del 2% que el modelo provee,
genera €6/día = €180/mes de edge bruto. Un precio de €29/mes es el 16% de ese valor → defensible.

---

## Tiers

### 🆓 Free — Track Record Público
- Acceso de solo lectura al track record histórico (todas las predicciones pasadas)
- Métricas de calibración y reliability diagram
- Sin predicciones del día en curso
- **Objetivo:** generar confianza antes del pago. El track record verificable ES el argumento de venta.

---

### 💡 Starter — €29 / mes
- Predicciones del día (Win Probability + E[R] interval P10–P50–P90)
- War Room completo para un equipo/día
- Análisis de divergencias (modelo vs manager)
- Acceso API no disponible
- **Target:** bettor individual con volumen bajo, grupos de picks amateurs

---

### 🚀 Pro — €79 / mes
- Todo lo de Starter +
- Todos los partidos del día (no solo un equipo)
- Modelo de fatiga (cuando esté implementado — Roadmap 3.1)
- Export CSV de predicciones diarias
- CLV tracker (cuando esté implementado — Roadmap 3.2)
- Alertas de valor cuando P(W) > 60% con IC90 bien definido
- **Target:** bettor profesional, grupos de picks activos, traders

---

### 🏢 Enterprise / API — precio a negociar
- API REST con todas las predicciones en tiempo real
- SLA de disponibilidad, soporte dedicado
- White-label para sportsbooks o plataformas de picks
- Historial completo + backfill
- **Target:** sportsbooks LatAm, operadores DFS, plataformas de análisis

---

## Lo que desbloquea cada nivel

| Feature | Free | Starter | Pro | Enterprise |
|---------|------|---------|-----|-----------|
| Track record histórico | ✓ | ✓ | ✓ | ✓ |
| Calibración / reliability diagram | ✓ | ✓ | ✓ | ✓ |
| Predicciones del día (1 equipo) | ✗ | ✓ | ✓ | ✓ |
| Todos los partidos del día | ✗ | ✗ | ✓ | ✓ |
| Modelo de fatiga | ✗ | ✗ | ✓ | ✓ |
| CLV tracker | ✗ | ✗ | ✓ | ✓ |
| Export CSV | ✗ | ✗ | ✓ | ✓ |
| API access | ✗ | ✗ | ✗ | ✓ |
| White-label | ✗ | ✗ | ✗ | ✓ |

---

## Modelo de facturación

- Mensual (sin contrato) para Free/Starter/Pro
- Anual con descuento del 20% para Starter/Pro
- Enterprise: contrato anual, facturación mensual o trimestral

## Moneda y pagos

- EUR para España/Europa
- USD para LatAm y mercado hispano US
- Stripe para procesamiento; posiblemente MercadoPago para LatAm en segunda fase
