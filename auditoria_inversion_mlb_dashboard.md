# AUDITORÍA DE INVERSIÓN — MLB ANALYTICS DASHBOARD
**Clasificación:** Due Diligence Nivel Senior  
**Estándar de referencia:** Goldman Sachs Data Intelligence · McKinsey Analytics · Palantir-grade Decision Tools  
**Fecha de auditoría:** 27/05/2026  
**Módulos auditados:** War Room · Post-Game Review · Track Record

---

> **NOTA DE METODOLOGÍA:** Esta auditoría no busca puntos positivos. Identifica cada vector de riesgo que bloquearía una decisión de inversión Seed/Series A en un producto de datos deportivos. El estándar de comparación es Zelus Analytics, Sports Info Solutions (SIS) o Statcast Enterprise.

---

## MÓDULO 1 — WAR ROOM (Pre-Game Intelligence)

### 1. DIAGNÓSTICO CRÍTICO

**1.1 — Intervalo de Confianza IC90 Estadísticamente Incoherente**
El IC90 de probabilidad de victoria muestra `35%–37%`, un rango de **2 puntos porcentuales** para un evento binario en un deporte de alta varianza como el béisbol. Esto es una anomalía estadística severa: un modelo MonteCarlo con 10.000 simulaciones sobre un partido de béisbol producirá bandas de incertidumbre de 15–25 puntos porcentuales como mínimo en mercados competitivos.

**1.2 — P10 de Carreras Esperadas = 0.0**
El percentil 10 de carreras esperadas se reporta como `0.0`. Ningún equipo de la MLB ha terminado un partido con 0 carreras de forma sistemática; la media histórica del equipo perdedor ronda 2.8 carreras. Este valor indica un fallo en la distribución de la cola izquierda del modelo, probablemente una distribución de Poisson sin truncamiento inferior adecuado o un error en la visualización del intervalo.

**1.3 — What-If Simulator Completamente Inoperativo**
Los tres KPIs del simulador muestran `ΔE[R] = -0.000`, `ΔP(W) = -0.0pp`, `ΔWOBA = est.`. Cuando el output principal de una funcionalidad de alto valor es cero en todos los ejes y "est." (no calculado) en el tercero, el módulo no está funcionando. Esto es un bloqueante crítico: el simulador what-if es la funcionalidad de diferenciación más valiosa del producto para equipos técnicos.

**1.4 — Indicador "HIGH" de Incertidumbre Sin Accionabilidad**
El banner `HIGH` de incertidumbre aparece en rojo sin ninguna guía operacional: ¿qué debe hacer el usuario cuando la incertidumbre es alta? ¿Reducir exposición? ¿Aumentar el umbral de confianza? La alerta existe pero no lleva a ninguna decisión. Es decorativa.

**1.5 — Ausencia de Contexto Histórico del Matchup**
El panel muestra el ERA del pitcher (`4.25`) pero no el rendimiento histórico de ese pitcher específico contra este lineup, ni el splits LHB/RHB en el contexto del estadio actual. El panel "Ventajas/Riesgos" existe pero es genérico (OBP, ISO) y no contextualiza las métricas contra el pitcher específico del día.

**1.6 — Métricas del Lineup Sin Jerarquía Visual de Impacto**
La tabla de lineup muestra AVG, OPS, wOBA, OBP, ISO para 9 jugadores simultáneamente. Son 45 celdas de datos sin jerarquía visual que indique cuáles métricas son accionables para este matchup específico. Un directivo técnico necesita saber en 3 segundos qué es relevante; aquí necesita leer toda la tabla.

**1.7 — "Optimización Global" Sin Explicación del Algoritmo**
El botón "Optimización Global" activa una reordenación del lineup pero no muestra qué función objetivo está maximizando (¿E[R]? ¿wRC+? ¿probabilidad de anotar en la primera entrada?). Un black box en un producto de decisiones críticas destruye la confianza institucional.

**1.8 — Precisión 30d del 74% Sin Denominador Contextualizado**
`74% | 847 partidos` aparece como credencial de precisión pero sin especificar: ¿precisión en qué métrica? ¿En dirección de ganador? ¿En over/under de carreras? ¿Con qué umbral de confianza mínima? Un 74% de accuracy en clasificación binaria sin contexto puede ser trivialmente inferior a predecir siempre al favorito de Vegas.

---

### 2. EL 'PORQUÉ' — IMPACTO DE NEGOCIO

| Fallo | Impacto en Decisión de Inversión |
|---|---|
| IC90 de 2pp | Destruye credibilidad del modelo ante cualquier estadístico. Primera pregunta en un pitch: "¿Cómo obtienen un intervalo de 2pp en béisbol?" |
| P10 = 0.0 | Señal de que la validación de outputs del modelo no existe. Red flag para due diligence técnica. |
| Simulator = 0 | La feature de mayor valor percibido no funciona. Si esto llega a una demo, la reunión termina. |
| HIGH sin acción | Un indicador de riesgo sin protocolo de respuesta no es inteligencia; es ruido. No diferencia el producto de una hoja de Excel. |
| Métricas sin jerarquía | Impide la adopción por usuarios no expertos (GMs, coaches). Limita el mercado total direccionable (TAM). |

---

### 3. PLAN DE ACCIÓN — REMEDIACIÓN

**3.1 — IC90 Incoherente**
Revisar el cálculo del IC90 para probabilidad de victoria. El IC90 debe derivarse de la distribución de P(Victoria) a través de las 10.000 simulaciones MonteCarlo, no del intervalo de la estimación puntual. Implementar bootstrap sobre los resultados de simulación: `np.percentile(sim_results, [5, 95])`. Mostrar el resultado histórico: en béisbol, esperar bandas de ±12–18pp.

**3.2 — P10 = 0.0**
Implementar truncamiento en la distribución de carreras esperadas. Usar una distribución Negativa Binomial o Zero-Inflated Poisson con floor en 1 carrera por equipo, o al menos mostrar el P5 real de la distribución empírica histórica. Validar el cálculo de percentiles contra distribuciones históricas de MLB (2015–2025).

**3.3 — Simulator What-If**
Prioridad 0: el simulador debe recalcular E[R], P(W) y wOBA proyectada cuando se realizan cambios en el lineup. Arquitectura sugerida: cada modificación de lineup dispara una nueva simulación parcial (1.000 iteraciones, no 10.000) con el lineup modificado, devolviendo el delta respecto al baseline en menos de 200ms. Mostrar siempre el valor absoluto además del delta.

**3.4 — Indicador HIGH**
Añadir un tooltip expandible al indicador de incertidumbre con tres opciones de accionabilidad estandarizadas:
- **HIGH:** "Aumentar umbral de confianza mínima al 60% antes de actuar."
- **MEDIUM:** "Usar predicción con reserva. Validar con intel adicional."
- **LOW:** "Confianza estructural suficiente. Proceder con lineup óptimo."

**3.5 — Contexto del Matchup**
Añadir sub-panel "Pitcher vs. este Lineup": AVG histórico del lineup actual contra este pitcher (o pitchers con ERA/FIP similares), número de ABs como muestra, y xFIP ajustado al parque del día.

**3.6 — Jerarquía Visual del Lineup**
Colapsar la tabla de 5 métricas a 2 métricas primarias configurables por el usuario (default: wOBA y OPS). Añadir un "relevance score" por jugador que pondera las métricas según el matchup específico (ej: contra RHP con K% alto, ISO pesa más que OBP).

**3.7 — Transparencia del Algoritmo**
Añadir un botón "¿Cómo funciona?" junto a "Optimización Global" que muestre en un modal: función objetivo (ej: maximizar E[R] en las primeras 6 entradas), restricciones (posición, lado del bate), y el impacto marginal de cada swap.

**3.8 — Contextualización de la Precisión**
Desglosar la métrica de precisión en: `Accuracy (ganador)`, `MAE (carreras)`, `Brier Score (prob. victoria)`. Añadir un benchmark externo visible: "vs. línea de Vegas" o "vs. modelo base ELO". Sin benchmark, el 74% no tiene valor de referencia.

---

## MÓDULO 2 — POST-GAME REVIEW (Análisis Histórico)

### 1. DIAGNÓSTICO CRÍTICO

**2.1 — Error de Predicción de +9.59 Carreras Sin Diagnóstico de Causa**
El modelo predijo `E[R] = 5.41` para un partido que terminó `15-6` (21 carreras totales). El error neto es `+9.59 carreras vs. real`. La pantalla lo muestra como métrica pero **no ofrece ninguna explicación causal**: ¿fue error del modelo de pitcher? ¿Un inning de explosión atípica? ¿Condiciones de viento no capturadas? Sin análisis de causa raíz, el dato es un acusador sin juicio.

**2.2 — "56.6% Win Probability" en un Partido 15-6 es Inaceptable**
El modelo asignó solo 56.6% de probabilidad al equipo local (Dodgers) en un partido que terminaron ganando 15-6. Esto indica que el modelo no está capturando asimetrías estructurales del partido (home team, lineup power, pitcher ERA gap). Un modelo bien calibrado debería haber asignado 70–80%+ a Los Ángeles en este contexto.

**2.3 — Diagrama de Calibración con Datos Insuficientes**
El Reliability Diagram muestra `0 predicciones / datos de entrenamiento` en el comentario y la curva de calibración tiene muy pocos puntos de datos visibles. Un diagrama de calibración requiere mínimo 1.000 predicciones agrupadas en deciles para ser estadísticamente interpretable. Con menos datos, la curva oscila por azar y no valida ni invalida el modelo.

**2.4 — Brier Score "Por Debajo de la Media"**
El Brier Score temporal muestra explícitamente `Por debajo de la media`. Esto significa que el modelo de probabilidad de victoria está rindiendo peor que el benchmark de la liga. Este es el dato más crítico de toda la pantalla y está presentado con el mismo peso visual que otros indicadores secundarios. En un producto que vende precisión predictiva, esto es un hallazgo que requiere una sección de análisis dedicada, no un badge de color.

**2.5 — Análisis de Divergencias Sin ROI Calculado**
El módulo de divergencias lineup identifica 2 decisiones "significativas" del manager, pero no cuantifica el impacto real de esas decisiones en el resultado del partido. "Manager mejor en slot 2" y "Coste al equipo en slot 5" son etiquetas sin evidencia post-hoc: ¿cuántas carreras costó esa decisión según el modelo?

**2.6 — IC90 Lineal en Distribución de Carreras Asimétrica**
La visualización de E[R] proyectado usa un intervalo simétrico (P10–P90 como barras paralelas) cuando la distribución de carreras en béisbol es altamente asimétrica a la derecha (cola larga positiva). Un juego de 21 carreras totales debería haber estado en el P90+ del modelo; la representación visual no captura esta asimetría.

---

### 2. EL 'PORQUÉ'

| Fallo | Impacto |
|---|---|
| Error +9.59 sin diagnóstico | El producto no aprende de sus propios errores. Señal de ausencia de feedback loop técnico. |
| 56.6% en partido 15-6 | Modelo de clasificación débilmente calibrado. Cualquier evaluador técnico lo detectará inmediatamente. |
| Calibration con N insuficiente | La métrica de fiabilidad más importante no es interpretable. Equivale a no tener métrica de calibración. |
| Brier "por debajo de la media" | El KPI central del producto está en rojo. Ningún inversor capitaliza un modelo que rinde por debajo del benchmark. |
| Divergencias sin ROI | El análisis de decisiones es opinión, no evidencia. Reduce la confianza del usuario experto (analistas, GMs). |

---

### 3. PLAN DE ACCIÓN — REMEDIACIÓN

**2.1 — Análisis de Causa Raíz del Error**
Implementar un módulo de "Error Attribution" que descomponga automáticamente los errores grandes (>4 carreras) en categorías: (a) Error de modelo de pitcher (xFIP vs FIP real), (b) Eficiencia ofensiva atípica (LOB% vs esperado), (c) Innings de explosión (>4 carreras en 1 inning), (d) Factor climático no capturado. Esto convierte el error en conocimiento accionable para el siguiente partido.

**2.2 — Recalibración del Modelo de Clasificación**
Aplicar calibración de Platt Scaling o Isotonic Regression sobre las probabilidades de salida del modelo. Comparar las curvas de calibración antes/después con un conjunto de validación de al menos 500 partidos. El objetivo es que cuando el modelo dice 70%, el equipo local gane el 70% de las veces, no el 56%.

**2.3 — Umbral Mínimo de Datos para el Reliability Diagram**
No renderizar el diagrama de calibración hasta tener N ≥ 500 predicciones con resultado. Mientras tanto, mostrar: "Calibración en construcción: X/500 partidos requeridos." Mostrar el diagrama preliminar con intervalos de confianza (error bars por decil usando Wilson score interval).

**2.4 — Elevar el Brier Score a KPI Principal**
Rediseñar el layout del módulo de evaluación: el Brier Score debe ocupar 40% del espacio visual con comparación temporal (últimos 30 / 90 / temporada completa) y una trayectoria de mejora. Si está "por debajo de la media", el panel debe mostrar el plan de mejora activo, no solo el badge rojo.

**2.5 — ROI de Divergencias**
Para cada divergencia significativa, calcular la diferencia de E[R] entre el lineup del modelo y el lineup real usado. Al final del partido, comparar contra el resultado y añadir: "Si el manager hubiera seguido el modelo en el slot 5, el E[R] adicional habría sido +0.027 carreras (dentro del ruido estadístico)" o "...habría sido +0.15 carreras (relevante)."

**2.6 — Distribución Asimétrica**
Reemplazar la visualización de intervalo simétrico por un violin plot o distribución de densidad del kernel (KDE) de las carreras simuladas. Marcar el resultado real en el eje X. Esto permite al usuario ver visualmente si el resultado fue "dentro del modelo" aunque estuviera en la cola.

---

## MÓDULO 3 — TRACK RECORD (Registro de Predicciones)

### 1. DIAGNÓSTICO CRÍTICO

**3.1 — "Backtest no generado aún" — Bloqueante Absoluto**
El banner de advertencia al inicio de la pantalla indica que el backtest no ha sido ejecutado. El Track Record es la prueba de concepto más crítica para cualquier evaluador externo. Sin backtest, no hay evidencia histórica verificable del rendimiento del modelo. Esto descalifica el módulo completo.

**3.2 — Sin Métricas Agregadas en el Header**
La pantalla abre directamente en una tabla de filas sin ningún KPI agregado visible: no hay MAE total, no hay % de predicciones correctas por umbral, no hay P&L acumulado, no hay Brier Score de temporada. El usuario tiene que derivar mentalmente el rendimiento del modelo a partir de cientos de filas individuales. Esto no es un producto; es un log de datos.

**3.3 — Columna "Correcto" Vacía en Datos Recientes**
Las predicciones del `2026-05-27` (día actual) muestran la columna "Correcto" vacía, lo cual es esperable si los partidos no han terminado. Pero no hay ninguna señal visual que distinga entre "partido en curso" (sin resultado aún), "partido finalizado con predicción correcta" y "partido finalizado con predicción incorrecta". Las tres categorías se ven idénticas en la tabla.

**3.4 — Nomenclatura de Columnas Ambigua**
La columna "AciRi" no tiene denominación estándar en ningún framework de evaluación de modelos conocido. No es ni RMSE, ni MAE, ni log-loss. Sin documentación inline (tooltip, leyenda), el usuario externo no puede interpretar esta métrica, lo que destruye la verificabilidad del track record.

**3.5 — Ausencia de Filtros por Relevancia del Partido**
La tabla mezcla predicciones de partidos con alta incertidumbre y baja incertidumbre sin distinción. Un track record profesional debe permitir filtrar por: nivel de confianza del modelo (>60%, >70%, >80%), tipo de apuesta implícita (over/under, ganador), y rango de E[R] predicho. Sin esto, el track record no es auditable.

**3.6 — Sin Curva de Performance Acumulada**
No existe ninguna visualización de la evolución del rendimiento del modelo a lo largo del tiempo. Una tabla de 100+ filas sin un gráfico de P&L acumulado o Brier Score rolling no permite detectar si el modelo está mejorando, degradándose, o tiene sesgos estacionales.

**3.7 — Sin Separación entre Datos In-Sample y Out-of-Sample**
El track record no distingue visualmente qué predicciones fueron generadas en tiempo real (verdadero out-of-sample) versus las que podrían haber sido generadas post-hoc. Para un inversor técnico, esta distinción es fundamental: cualquier backtest sin esta separación se asume como potencialmente contaminado por look-ahead bias.

---

### 2. EL 'PORQUÉ'

| Fallo | Impacto |
|---|---|
| Sin backtest | El producto no puede ser auditado. Ningún inversor firma un cheque sobre rendimiento no verificado. |
| Sin KPIs agregados | Imposible evaluar el producto en una primera reunión. El pitch fallaría en los primeros 60 segundos. |
| "AciRi" no definido | La métrica central del track record es opaca. Señal de falta de rigor metodológico. |
| Sin curva acumulada | No hay forma de detectar overfitting temporal o degradación del modelo. |
| Sin separación in/out-sample | Riesgo de look-ahead bias no descartado. Descalifica el track record en due diligence técnica. |

---

### 3. PLAN DE ACCIÓN — REMEDIACIÓN

**3.1 — Prioridad 0: Ejecutar Backtest**
Ejecutar `python backtest.py` sobre el conjunto histórico completo (mínimo 3 temporadas MLB: 2022–2024). Documentar metodología: walk-forward validation, no cross-validation simple. El backtest debe separar explícitamente el conjunto de entrenamiento y el de validación por año.

**3.2 — Header de KPIs Agregados**
Añadir un panel de resumen al inicio de Track Record con: MAE (carreras), Brier Score, % Over/Under correct, Log-Loss, y comparación contra benchmark Vegas Line. Estos KPIs deben ser el primer elemento visual de la pantalla, no la tabla.

**3.3 — Estados de Partido en la Tabla**
Añadir una columna "Estado" con tres valores: `🔴 En curso`, `✅ Correcto`, `❌ Incorrecto`. Usar iconografía consistente, no solo color (accesibilidad). Filtrar por estado desde el header de la tabla.

**3.4 — Renombrar y Documentar "AciRi"**
Reemplazar "AciRi" por la denominación estándar de la métrica que representa. Si es el error absoluto de carreras: llamarla `|ΔCarreras|`. Si es un score compuesto, documentarlo en un tooltip con la fórmula exacta. Publicar la metodología en una página de documentación enlazada desde el dashboard.

**3.5 — Sistema de Filtros**
Implementar filtros encadenables: `Confianza del modelo` (umbral de IC90), `Tipo de partido` (home/away, interliga), `Rango de E[R]`, `Resultado real (alta/baja puntuación)`. Añadir un botón "Solo con resultado" activo por defecto para no confundir predicciones pendientes con errores.

**3.6 — Gráfico de Performance Acumulada**
Añadir encima de la tabla un gráfico de línea temporal con: (a) Brier Score rolling 30 días, (b) MAE rolling 30 días, (c) Benchmark Vegas rolling 30 días. Esto es la visualización más importante del producto para un evaluador externo.

**3.7 — Etiquetado In-Sample / Out-of-Sample**
Añadir una columna o badge `OOS` (Out-of-Sample) a cada predicción generada en tiempo real. Las predicciones del backtest deben tener badge `BT` (Backtest) con color diferenciado. Añadir un filtro para ver solo predicciones OOS, que son las únicas válidas para auditoría.

---

## PROPUESTAS DE VALOR AÑADIDO

Estas funcionalidades elevarían el producto de "dashboard de visualización" a "plataforma de inteligencia competitiva" comparable a Zelus Analytics o Sports Info Solutions:

**A — Score de Valor de Mercado en Tiempo Real**
Comparar automáticamente el E[R] del modelo con las líneas de Las Vegas (API de The Odds API o Sportradar Odds). Cuando el modelo diverge >0.5 carreras del consenso de mercado, señalarlo como "Edge detectado" con el tamaño del edge cuantificado. Esto es el producto que realmente compran los operadores de franquicias y analistas de apuestas institucionales.

**B — Degradación de Modelo por Pitcher (Fatiga / Clima)**
Añadir un módulo que monitorice el rendimiento del modelo cuando el pitcher titular es reemplazado. El modelo actual parece estático respecto a la gestión del bullpen. Integrar datos de pitch count, días de descanso y splits recientes del bullpen para actualizar E[R] en tiempo real durante el partido.

**C — Comparación de Decisiones del Manager vs. Modelo (Serie Histórica)**
Para cada manager de la MLB, construir un perfil histórico de divergencias: "Este manager sigue el modelo óptimo en el 67% de los partidos. Sus divergencias tienen un impacto neto de -0.12 E[R] por partido." Esto convierte el Post-Game Review en un producto de scouting de managers con valor diferencial claro.

**D — API Exportable para Integración Institucional**
Los equipos, medios y operadores institucionales necesitan consumir los datos del modelo vía API, no solo vía dashboard. Ofrecer endpoints REST/WebSocket para E[R], P(Victoria) e IC90 en tiempo real con autenticación por API key. Esto multiplica el TAM sin aumentar el costo marginal de distribución.

**E — Alertas Push de Edge Detection**
Notificaciones automáticas (Slack webhook, email, push móvil) cuando el modelo detecta una divergencia significativa entre su predicción y el consenso de mercado, o cuando un evento en tiempo real (pitcher lesionado, cambio de lineup) mueve el E[R] más de 1 carrera. Convierte el producto pasivo en una herramienta activa de toma de decisiones.

**F — Backtesting de Estrategias de Lineup**
Permitir al usuario simular históricamente qué habría pasado si un manager hubiera seguido el modelo óptimo en todos los partidos de las últimas 3 temporadas. Cuantificar el impacto en W/L record y carreras anotadas. Este feature tiene valor de marketing directo para pitch a equipos de la MLB.

---

## VEREDICTO DE INVERSIÓN

> **No invertiría en el estado actual.**

El producto muestra una dirección técnica ambiciosa y coherente — la arquitectura de tres módulos (pre-game, post-game, track record) es la correcta — pero presenta fallos de validación estadística que ningún evaluador técnico serio ignoraría: un backtest sin generar, un simulador what-if inoperativo, un Brier Score por debajo del benchmark de mercado, y un IC90 estadísticamente incoherente en el módulo central. Invertiría con convicción si el equipo demuestra en 60 días: (1) backtest walk-forward de 3 temporadas con separación in/out-sample documentada, (2) Brier Score >= benchmark Vegas en el conjunto de validación, y (3) el simulador what-if funcionando con latencia < 200ms.

---

*Auditoría elaborada bajo estándar de Due Diligence Técnica para inversión en SaaS de datos deportivos. Versión 1.0 — 27/05/2026.*
