# Beachhead — Cliente Inicial (Roadmap 4.1)

> **Decisión:** elegir uno de los tres posibles clientes antes de desarrollar más.
> **Fecha:** 2026-05-27

---

## Los tres candidatos

### Opción A: Apuestas deportivas / DFS (Daily Fantasy Sports)
| Criterio | Evaluación |
|----------|-----------|
| TAM | ~$50–100B mercado legal de apuestas en EE. UU. y LatAm creciendo |
| Disposición a pagar | Alta — el edge vale directamente en dólares (CLV = retorno medible) |
| Facilidad de penetración | Media — DFS companies (DraftKings, FanDuel) tienen equipos internos; sportsbooks LatAm son más accesibles |
| Competencia | Moderada — pocos proveedores de analytics especializados en MLB para LatAm |
| Métrica de valor | CLV (Closing Line Value): ¿las predicciones baten la línea de cierre? |

**Clientes concretos identificados:**
1. Sportsbooks LatAm: Bet365 LatAm, Codere, Betano (Colombia, México, Argentina)
2. DFS operators: Underdog Fantasy, PrizePicks (mercado US)
3. Traders individuales con cuenta en books (B2C: suscripción mensual)
4. Grupos de picks profesionales (syndicates)
5. Plataformas de picks (PickleApp, Scores and Odds)

---

### Opción B: Media / Broadcast
| Criterio | Evaluación |
|----------|-----------|
| TAM | Limitado — pocos canales con presupuesto para analítica de lineup |
| Disposición a pagar | Baja-media — sport media tiene márgenes apretados |
| Facilidad de penetración | Alta si hay relación previa; difícil en frío |
| Métrica de valor | Engagement de audiencia, infografías dinámicas |

**Clientes concretos:** ESPN Deportes, Fox Sports LatAm, TUDN

---

### Opción C: Equipos MLB / Minor League
| Criterio | Evaluación |
|----------|-----------|
| TAM | Exactamente 30 equipos (mercado minúsculo) |
| Disposición a pagar | Alta — pero ciclos de venta de 12–24 meses |
| Facilidad de penetración | Muy baja — todos tienen departamentos de analítica propios |
| Competencia | Altísima — Statcast + Baseball Savant + equipos internos |
| Métrica de valor | $/WAR adicional; reducción de carreras cedidas |

---

## Decisión recomendada: **Opción A — Apuestas/DFS**

**Justificación:**
1. El edge del modelo es **directamente monetizable** para un bettor: CLV > 0 = dinero real.
2. El mercado LatAm de apuestas online creció ~40% CAGR 2022–2026 y está menos saturado que el US.
3. El producto actual (probabilidad de victoria + E[R] calibrado) **ya tiene la métrica correcta** para este segmento.
4. El español como idioma del producto es una **ventaja** en LatAm, no una debilidad.
5. El ciclo de venta es corto (suscripción B2C: días; B2B sportsbook: 2–4 meses vs. 12–24 de los equipos).

**Beachhead exacto:** traders individuales y grupos de picks en mercados LatAm (México, Colombia, Argentina, España) con acceso a books que ofrecen MLB.

---

## Próximos pasos

1. Implementar métrica CLV (Roadmap 3.2 — opción apuestas)
2. Redactar posicionamiento en español orientado a bettors (Roadmap 4.2)
3. Definir tier de entrada a €29/mes con acceso al modelo vía Track Record público
