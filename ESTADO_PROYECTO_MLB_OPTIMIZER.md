# Estado del Proyecto MLB Lineup Optimizer: Auditoría y Siguientes Pasos

> **Versión:** 2.0 — Mayo 2026
> **Clasificación:** Uso interno — Revisión técnica y gerencia deportiva
> **Alcance:** Sistema completo desde ingesta Statcast hasta recomendación táctica en el dugout

---

## Resumen Ejecutivo

El MLB Lineup Optimizer es un sistema de inteligencia artificial de producción que genera recomendaciones óptimas de alineación para el cuerpo técnico antes de cada partido. Combina datos históricos de Statcast, un modelo predictivo de resultados de turno al bate (LightGBM calibrado), un motor de simulación Monte Carlo (100 000 iteraciones), un algoritmo genético de optimización y un sistema RAG (Retrieval-Augmented Generation) con Claude para producir no solo la alineación matemáticamente óptima, sino también la justificación táctica que el mánager necesita para confiar en ella. El sistema está en producción con seis subsistemas de soporte activos: scorecard de confianza, detección de crisis, retroalimentación cualitativa del bench coach, protocolo de experimento A/B, **registro automático de divergencias** y un **bloque de automatización MLOps completo** (promoción de modelos, monitorización de disponibilidad, ingesta RAG, cierre de experimentos y reporte ejecutivo mensual).

---

## PARTE 1: Auditoría del Estado Actual (Estado del Arte)

### 1.1 Descripción Funcional: El Flujo Completo del Sistema

El sistema opera en tres capas temporales bien diferenciadas: datos históricos (días/semanas antes), inferencia pre-partido (horas antes del primer pitch) y soporte en tiempo real (durante el partido). A continuación se describe el flujo completo:

#### Capa 0 — Ingesta y Preparación de Datos (Pipeline Nocturno)

```
Statcast API (pybaseball)
         ↓
    Bronze Delta Lake
    (plate_appearances, pitcher_stats, weather)
         ↓
    Silver Delta Lake
    (datos limpios, validados por Great Expectations)
         ↓
    Gold Delta Lake / Parquet
    (feature matrix lista para entrenamiento e inferencia)
```

El pipeline de ingesta (`src/ingestion/statcast_ingestion.py`) descarga automáticamente los datos del día anterior tras cada noche de juegos. La capa de calidad (`src/quality/data_quality_rules.py`) valida integridad referencial, rangos de velocidad de salida (50–120 mph), proporciones de eventos y ausencia de fugas de datos futuras. Los datos meteorológicos se enriquecen vía `src/weather/weather_lambda.py` con temperatura, humedad y velocidad del viento del estadio.

#### Capa 1 — Feature Engineering (Ejecución bajo demanda o nocturna)

El motor `RollingFeaturesEngine` (`src/features/features_rolling.py`) computa 15+ features con ventanas deslizantes anti-leakage:

| Feature | Ventana | Descripción |
|---------|---------|-------------|
| `xwoba_7d / 15d / 30d` | 7, 15, 30 días | Expected wOBA reciente del bateador |
| `launch_speed_7d / 15d / 30d` | 7, 15, 30 días | Velocidad de salida media (mph) |
| `xwoba_ewma_alpha02 / 05` | EWMA | Promedio ponderado exponencial (responde más rápido a rachas) |
| `k_rate_7d / 30d` | 7, 30 días | Tasa de ponches |
| `bb_rate_7d / 30d` | 7, 30 días | Tasa de bases por bolas |
| `hard_hit_rate_30d` | 30 días | Tasa de contacto sólido (≥95 mph EV) |
| `babip_30d` | 30 días | Batting Average on Balls In Play |
| `hr_rate_30d` | 30 días | Tasa de jonrones |

Los splits de platoon (bateador zurdo vs. diestro según mano del lanzador rival) se calculan en `src/features/features_platoon.py`. Los embeddings de jugador (`src/embeddings/player_embeddings.py`) capturan similitudes históricas entre bateadores en espacio vectorial.

#### Capa 2 — Modelo Predictivo de Turno al Bate

El `AtBatPredictor` (`src/models/model_at_bat.py`) es un clasificador LightGBM con calibración isotónica (`CalibratedClassifierCV`) entrenado sobre ~340 features. Produce un **prob_vector de 7 clases** por cada combinación bateador × lanzador × contexto:

```
[P(OUT_IN_PLAY), P(STRIKEOUT), P(WALK_HBP), P(SINGLE), P(DOUBLE), P(TRIPLE), P(HOME_RUN)]
  suma = 1.0    vector calibrado    ECE ≤ 0.045
```

La calibración garantiza que "70% de probabilidad de contacto" sea estadísticamente honesto, no solo un score relativo. El umbral de calidad en producción es **Log-Loss ≤ 0.84** y **ECE ≤ 0.045**.

#### Capa 3 — Optimización de Alineación (Solicitud pre-partido)

```
POST /v1/optimize/lineup
          ↓
[OPCIONAL] Overrides cualitativos del bench coach
    (apply_feedback_overrides=True → escala prob_vectors)
          ↓
SabermetricSeeder → orden canónico inicial (semilla para el AG)
          ↓
Algoritmo Genético (GAConfig: 200 generaciones × 500 individuos)
    Fitness = E[R] estimado por simulación Monte Carlo rápida (5k runs)
          ↓
Top-K candidatos → Refinamiento Monte Carlo completo (100k runs)
    Resultado: best_lineup_indices, refinement_scores, E[R], Win%
          ↓
Detección de crisis (< 1ms, sin simulación adicional)
    Triggers: STAR_PLAYER_MAJOR_DROP | LEADOFF_OBP_SUBOPTIMAL | POWER_HITTER_BURIED
          ↓
[SI crisis] Reporte de Defensa RAG (Claude + Pinecone, ~10–15s)
          ↓
Scorecard de Confianza (stability × freshness × sample_coverage)
          ↓
LineupOptimizeResponse → API client
```

**Modos de operación:**

| Modo | Simulaciones | Latencia típica | Uso recomendado |
|------|-------------|-----------------|-----------------|
| `fast_mode=True` | 5 000 | ~3s | Exploración rápida, polling UI |
| `fast_mode=False` | 100 000 | ~25s | Decisión final pre-partido |

#### Capa 4 — Sistema de Confianza y Explicabilidad

El `LineupScorecard` presenta al mánager un diagnóstico de tres dimensiones:

- **Estabilidad de optimización** (55% del score): ¿Hay una brecha clara entre la mejor y la segunda mejor alineación? Brecha grande = alta confianza.
- **Frescura de datos** (25%): Horas desde la última actualización de features. Datos de hace 2h = alta confianza; 18h = baja confianza.
- **Cobertura de muestra** (20%): ¿Tiene el modelo suficientes plate appearances históricas del bateador vs. la mano del lanzador de hoy?

El score compuesto produce un semáforo: **HIGH** (≥0.80) → "Confía — desviarte cuesta ~0.3–0.5 carreras de expectativa"; **MEDIUM** (0.60–0.79) → "Revisa"; **LOW** (<0.60) → "Tu experiencia manda".

#### Capa 5 — Retroalimentación y Aprendizaje Continuo

El sistema captura retroalimentación en tres modalidades:

1. **Pre-partido** (`POST /v1/feedback/game`, `submission_type="pre_game"`): El bench coach reporta flags cualitativos (molestia física, distracción mental, racha caliente) con severidad `soft`/`hard` y peso [-1.0, +1.0]. Estos modifican los `prob_vector` antes de la optimización si se activa `apply_feedback_overrides=True`.

2. **Post-partido** (`submission_type="post_game"`): El cuerpo técnico evalúa la utilidad de la recomendación (escala 1–5) y registra contexto oculto que el sistema no vio.

3. **Experimento A/B — registro automático de divergencias** (`POST /v1/lineup/confirm`): Cuando el mánager confirma su alineación real, el endpoint compara automáticamente contra la recomendación AI cacheada de la última llamada a `/v1/optimize/lineup` y, si difiere, registra la observación A/B sin intervención adicional. La clasificación (order_only / player_swap / full_override) y el E[R] AI se capturan de la caché. Es idempotente: la segunda confirmación del mismo partido no duplica la observación. El registro manual vía `POST /v1/experiment/record-divergence` sigue disponible como fallback. Tras el partido, el cierre de la observación con el resultado real se realiza automáticamente mediante `PostGameResolver` (ver bloque de automatización). El endpoint `GET /v1/experiment/summary` calcula empíricamente si el mánager tiende a superar o quedar por debajo de la expectativa del sistema.

#### Capa 6 — MLOps y Reentrenamiento Diario

El `GameDayRetrainingPipeline` (`src/mlops/game_day_retraining.py`) ejecuta cada mañana:

```
1. Descarga datos Statcast del día anterior
2. Reconstruye features rolling
3. Entrena AtBatPredictor (challenger)
4. Evalúa en holdout: Log-Loss ≤ 0.84 ?
5. Compara con campeón en producción (MLflow champion/challenger)
6. Si mejora: registra como "Production_Candidate" + alias "staging"
7. Si falla: emite alerta estructurada, aborta sin tocar producción
```

La promoción final a `alias="production"` es ahora **automática** mediante `AutoPromoter` (ver Bloque de Automatización):

- Gate 1: Δlog-loss ≥ 0.01 (challenger debe mejorar al menos 0.01 sobre el campeón)
- Gate 2: ECE del challenger ≤ 0.035 (calibración dentro de tolerancia)
- Si ambos gates pasan: reasigna alias `production` en MLflow, etiqueta el campeón anterior como `retired` y despliega la nueva imagen Docker en ECS Fargate.

#### Bloque de Automatización MLOps (implementado en esta versión)

| Módulo | Archivo | Función |
|--------|---------|---------|
| Promotor automático | `src/mlops/auto_promoter.py` | Champion/challenger gate + despliegue ECS Fargate |
| Monitor de disponibilidad | `src/mlops/mlb_stats_monitor.py` | Poll MLB StatsAPI cada 15min → PostgreSQL + Redis cache invalidation |
| Lambda de scouting | `src/rag/lambda_scouting_trigger.py` | S3 ObjectCreated → Pinecone ingestion (AWS Lambda) |
| Resolver post-partido | `src/experiment/post_game_resolver.py` | Cierre automático de observaciones A/B via boxscore API (Airflow) |
| Reporte ejecutivo mensual | `src/reporting/automated_reporter.py` | PDF Plotly con 4 gráficos + distribución vía AWS SES |

**`AutoPromoter`** (`src/mlops/auto_promoter.py`): Evalúa challenger (alias `staging`) vs. campeón (alias `production`) usando dos gates: Δlog-loss ≥ `PROMOTER_DELTA_LOG_LOSS` (default 0.01) y ECE del challenger ≤ `PROMOTER_ECE_CEILING` (default 0.035). Si ambos pasan: reasigna el alias `production` en MLflow, etiqueta el campeón anterior como `retired` y llama a `boto3.client("ecs").register_task_definition()` + `update_service(forceNewDeployment=True)`. Raise `PromotionError` ante cualquier fallo no recuperable. Configurable via variables de entorno para ejecución como Airflow `PythonOperator` o CLI.

**`MLBStatsMonitor`** (`src/mlops/mlb_stats_monitor.py`): `APScheduler BackgroundScheduler` con trigger cada 15 minutos. Llama a `/teams/{id}/roster/40Man` y `/transactions` de la MLB Stats API, detecta cambios de disponibilidad (DL placements, activaciones) y ejecuta upsert batch en PostgreSQL (`player_availability`). Publica un mensaje Redis en el canal `optimizer:cache:invalidate` con los IDs de jugadores afectados para que el servicio de inferencia invalide caches de prob_vectors.

**`lambda_scouting_trigger`** (`src/rag/lambda_scouting_trigger.py`): Handler AWS Lambda que reacciona a eventos `S3:ObjectCreated`. Parsea la clave S3 bajo el patrón `scouting/{team}/{scout_type}/{player_id}_{name}_{date}.{ext}` (tipos válidos: `pro_scout`, `medical`, `developmental`, `analytics`). Extrae el texto (UTF-8 con fallback latin-1), construye un `ScoutReport` y llama a `ScoutingIngestionPipeline.ingest_report()` para indexar en Pinecone. En batches de 1 registro re-lanza la excepción (activa DLQ); en batches múltiples incrementa `failed` y continúa.

**`PostGameResolver`** (`src/experiment/post_game_resolver.py`): Job nocturno de Airflow. Para cada fecha, escanea las particiones Parquet del `ExperimentStore` buscando observaciones `status="open"`, consulta el schedule de la MLB Stats API para construir un mapa `abbrev → gamePk`, descarga el boxscore y llama a `experiment_store.record_outcome(home_runs)` para cerrar la observación. Devuelve métricas de éxito/error para el task de Airflow.

**`AutomatedReporter`** (`src/reporting/automated_reporter.py`): Genera un PDF de 4 páginas con: portada de KPIs, tendencia de adopción AI vs. manager, distribución de tipos de divergencia (pie chart) e indicador de precisión E[R]. Renderiza con kaleido, fusiona con pypdf y distribuye vía AWS SES (`send_raw_email` con adjunto MIME). Callable de Airflow: `generate_monthly_report(execution_date, send_email)`.

**Registro automático de divergencias** (`POST /v1/lineup/confirm`): Tras llamar a `/v1/optimize/lineup`, la recomendación AI queda cacheada en `_state.lineup_cache` con TTL de 24h. Cuando el mánager confirma su alineación real, el endpoint compara, clasifica y registra la divergencia automáticamente. Es idempotente (segunda llamada mismo `game_id` no duplica la observación; `observation_id = sha256(game_id)[:16]`). HTTP 404 si no hay caché para el `game_id` (no hubo llamada previa a `/optimize/lineup`). Nota de arquitectura: el caché es in-memory; para despliegues multi-worker externalizar a Redis.

---

### 1.2 Utilidad de Negocio

#### Impacto cuantificable para el cuerpo técnico

| Métrica | Sin sistema | Con sistema |
|---------|------------|-------------|
| Tiempo de análisis pre-partido | ~30–45 min (análisis manual) | < 3 minutos (fast_mode) |
| Precisión de win expectancy | Estimación intuitiva ±15% | Monte Carlo calibrado con IC explícito |
| Fundamentación de decisiones | "Sensación del mánager" | E[R] cuantificado + 3 razones top en español |
| Retención de conocimiento táctico | Memorias individuales del coach | FeedbackStore Parquet + embeddings de scouts en Pinecone |
| Detección de incoherencias | Post-mortem reactivo | Proactiva (crisis triggers < 1ms) |

#### Valor diferencial específico

- **Reducción de incertidumbre en splits de platoon:** El modelo tiene en cuenta la mano del lanzador y los históricos específicos de la matchup, eliminando sesgos cognitivos comunes (ej. sobreponderar el rendimiento reciente vs. el histórico).
- **Efecto compuesto de optimización:** La diferencia entre la mejor alineación y una alineación arbitraria es típicamente 0.15–0.35 E[R] por partido. Proyectado a 162 partidos: **24–57 carreras adicionales de expectativa por temporada**.
- **Reporte de defensa proactivo:** Cuando el sistema recomienda algo contraintuitivo, genera automáticamente el informe de justificación RAG antes de que el mánager lo pida. Esto elimina la resistencia al cambio más común ("¿por qué el sistema pone a X en el tercer slot?").
- **Ciclo de aprendizaje institucional:** A través del FeedbackStore, el conocimiento táctico del bench coach se convierte en datos de entrenamiento estructurados (sample weights para reentrenamiento, propuestas de features binarias).

---

### 1.3 Operaciones Manuales (Human-in-the-Loop)

Las siguientes acciones requieren intervención humana para garantizar la calidad del sistema. Las acciones marcadas con ✅ **AUTO** han sido automatizadas en esta versión. Las marcadas con 🔵 **PARCIAL** requieren supervisión humana solo ante fallos o alertas.

#### Producción del modelo

| Acción | Estado | Frecuencia / Cuándo | Responsable |
|--------|--------|---------------------|-------------|
| ~~Aprobación de nuevo modelo a producción~~ | ✅ **AUTO** — `AutoPromoter` evalúa Δlog-loss ≥ 0.01 y ECE ≤ 0.035; despliega en ECS automáticamente | Diario — tras reentrenamiento | Ninguno (solo si `PromotionError`) |
| Revisión de la comparativa campeón vs. retador (Log-Loss, ECE, calibración por clase) | 🔵 **PARCIAL** — los gates automáticos cubren casos normales | Semanal o ante `PromotionError` | Ingeniero ML |
| Análisis de model drift cuando Log-Loss supera 0.84 o ECE supera 0.045 | Manual | Semanal o al detectar degradación | Ingeniero ML |
| Validación de la cobertura de nuevas temporadas (features de expansión, rookies) | Manual | Al inicio de temporada (marzo) | Data Engineer + Ingeniero ML |

#### Pipeline de datos

| Acción | Estado | Frecuencia / Cuándo | Responsable |
|--------|--------|---------------------|-------------|
| Verificación de ingesta Statcast tras errores de API | Manual | Cuando falla el pipeline nocturno | Data Engineer |
| Validación de reglas Great Expectations con nuevas fuentes | Manual | Al añadir nuevas variables | Data Engineer |
| Backfill de datos históricos cuando Statcast corrige retroactivamente | Manual | ~2–3 veces por temporada | Data Engineer |
| ~~Actualización de rosters y lesiones~~ | ✅ **AUTO** — `MLBStatsMonitor` hace poll cada 15 min vía MLB StatsAPI; upsert en PostgreSQL + invalidación Redis automática | Continuo (APScheduler) | Ninguno (solo ante fallo del scheduler) |

#### Sistema RAG y conocimiento táctico

| Acción | Estado | Frecuencia / Cuándo | Responsable |
|--------|--------|---------------------|-------------|
| ~~Ingesta manual de informes de scouts a Pinecone~~ | ✅ **AUTO** — `lambda_scouting_trigger` reacciona a S3:ObjectCreated; solo hay que subir el archivo al bucket | Al subir el PDF/TXT a S3 | Analista de Scouting (solo sube el archivo) |
| Revisión de la calidad del Reporte de Defensa RAG (precisión táctica, alucinaciones) | Manual | Tras cada activación de crisis | Analista de Baseball + Ingeniero ML |
| Actualización del system prompt del explainer | Manual | Al cambiar el estilo de comunicación del staff | Analista de Baseball |

#### Retroalimentación cualitativa (bench coach)

| Acción | Estado | Frecuencia / Cuándo | Responsable |
|--------|--------|---------------------|-------------|
| Envío de flags pre-partido del bench coach (`POST /v1/feedback/game`) | Manual — requiere decisión humana | Diario — antes del primer pitch | Bench Coach / Asistente técnico |
| ~~Registro manual de divergencias~~ (`POST /v1/experiment/record-divergence`) | ✅ **AUTO** — `POST /v1/lineup/confirm` compara y registra automáticamente al confirmar la alineación real | Al confirmar la alineación | App del mánager (UI) |
| ~~Registro del resultado post-partido~~ (`POST /v1/experiment/record-outcome`) | ✅ **AUTO** — `PostGameResolver` descarga el boxscore y cierra la observación cada noche (Airflow) | Nocturno | Ninguno (solo ante fallo del job) |
| Revisión de propuestas de features recurrentes (Layer 3 del FeedbackStore) | Manual — requiere decisión humana | Semanal — `GET /v1/feedback/feature-proposals` | Ingeniero ML + Analista de Baseball |
| Aprobación e integración de features propuestas al pipeline | Manual — decisión de calidad | Quincenal | Ingeniero ML |

#### Estrategia y monitorización

| Acción | Estado | Frecuencia / Cuándo | Responsable |
|--------|--------|---------------------|-------------|
| ~~Preparación del informe ejecutivo A/B~~ | ✅ **AUTO** — `AutomatedReporter` genera PDF de 4 páginas con Plotly y lo distribuye por AWS SES (Airflow callable) | Mensual | Ninguno (solo configurar destinatarios en `ses_to_emails`) |
| Revisión del resumen A/B (`GET /v1/experiment/summary`) — ¿el mánager supera al AI? | Manual — decisión estratégica | Mensual | Director de Analytics |
| Ajuste de umbrales del scorecard (High/Medium/Low) según madurez del modelo | Manual | Trimestral | Ingeniero ML + Director de Analytics |
| Auditoría de Prometheus: latencia P99, tasa de errores, coverage | Manual | Semanal | Ingeniero de plataforma |

**Resumen de automatización**: 6 de las 18 operaciones manuales originales están ahora automatizadas (33%). Las 12 restantes requieren juicio humano genuino (decisiones de calidad de modelo, conocimiento táctico del bench coach, revisiones de contenido RAG).

---

## PARTE 2: Estrategia de Mejora (R&D)

### 2.1 Feature Engineering Profundo

El modelo actual captura con precisión la forma reciente del bateador y los splits de platoon estándar. Las siguientes dimensiones no explotadas representan el mayor potencial de ganancia de precisión:

#### A. Modelado del Estado del Lanzador Rival (más alto impacto)

El sistema actualmente recibe `era`, `pitch_mix` y `hand` del lanzador como inputs estáticos. La realidad del juego es dinámica:

- **Fatiga acumulada:** Un starter con 110 lanzamientos lanzados 4 días atrás y otros 30 hace 2 días tiene una capacidad de pitching radicalmente distinta a uno fresco. La feature `pitcher_pitches_last_5d` (rolling sum desde `statcast_pitcher`) captura esto directamente.
- **Desgaste del arsenal según el conteo:** Usando datos de Statcast pitch-by-pitch, se puede construir `pitcher_velocity_drop_5th_inning` (diferencia media en velocidad del fastball entre innings 1-2 e innings 5+) como indicador de desgaste intrapartido.
- **Rotación del bullpen:** La disponibilidad efectiva del bullpen (lanzadores que lanzaron >20 lanzamientos en los últimos 2 días) afecta la estrategia del mánager rival y, por extensión, cuándo el lineup enfrentará pitching de calidad inferior. Esta feature binaria `bullpen_depleted_flag` ya tiene campo en el sistema de feedback pero no está en el modelo.

#### B. Datos de Biomecánica y Tracking

Statcast publica datos de extensión de release point del lanzador, spin rate por tipo de lanzamiento y horizontal/vertical break. Estos datos permiten construir features de "dificultad de matchup" que van más allá de la mano del lanzador:

- `break_differential_vs_avg`: La diferencia en movimiento del slider del lanzador de hoy vs. el promedio de sliders que el bateador ha enfrentado en los últimos 30 días. Un bateador acostumbrado a sliders de poca ruptura contra un lanzador con ruptura extrema es una desventaja no capturada.
- `pitcher_arm_slot_percentile`: Ángulos de release inusuales (ej. submarina) son difíciles de ver para ciertos bateadores. Este percentil ya tiene correlación demostrada con BABIP.

#### C. Métricas de Contexto Situacional

- **Leverage Index dinámico por posición del batting order:** En condiciones reales, el 3er y 4o bateador enfrentan situaciones de mayor leverage. Incorporar `avg_leverage_by_slot_position` como feature durante el entrenamiento ayudaría al modelo a entender que el mismo jugador tiene impacto diferente según el slot.
- **Historial de bateador en el estadio específico:** Park factors estáticos son una aproximación; el historial individual del bateador en ese parque (con ajuste por muestra) captura sesgos como preferencia por el viento o dimensiones.
- **Temperatura real vs. historial del bateador:** Algunos bateadores tienen splits documentados fríos/calientes. La correlación entre temperatura actual y `xwoba_7d` del bateador puede capturarse como feature de interacción.

---

### 2.2 Modelado Avanzado

#### A. Mejora de la Calibración para Reducir ECE

El sistema actual alcanza ECE ≤ 0.045 con calibración isotónica. La siguiente frontera:

- **Calibración por subgrupo:** El ECE global de 0.045 puede enmascarar un ECE de 0.08 para el subgrupo "zurdo vs. lanzador zurdo con <30 PA históricos". Una calibración Platt por grupos de platoon (L vs L, L vs R, R vs L, R vs R) × rango de sample size eliminaría este sesgo estructural.
- **Conformal Prediction Intervals:** En lugar de devolver solo el prob_vector puntual, el sistema podría devolver intervalos de predicción conformales con cobertura garantizada al 90%. Esto traduce directamente a incertidumbre cuantificada en el scorecard: "P(HR) ∈ [3.2%, 8.1%]" en lugar de "P(HR) = 5.1%". El mánager vería qué tan ancha es la incertidumbre antes de decidir.
- **Recalibración online:** Un recalibrador ligero (Platt scaling con regularización L2) que se actualiza diariamente con los últimos 7 días de juego podría reducir el ECE en periodos de cambio de temporada o tras lesiones masivas.

#### B. Reducción de Varianza en la Simulación Monte Carlo

El motor actual usa 100k iteraciones sin varianza reducida. Dos mejoras técnicas con impacto directo en latencia y precisión:

- **Control Variates / Importance Sampling:** Usar el E[R] de la alineación canónica sabermétricamente óptima como variable de control reduce la varianza del estimador de E[R] en ~30–40%, permitiendo alcanzar la misma precisión con ~60k iteraciones (reducción de ~15s de latencia en modo full).
- **Quasi-Monte Carlo (Sobol sequences):** Reemplazar el muestreo aleatorio estándar por secuencias de baja discrepancia (Sobol/Halton) en la simulación de innings reduce el error estadístico en O(log(N)/N) vs. O(1/√N). Con 20k iteraciones se podría alcanzar la precisión actual de 100k.

#### C. Arquitectura del Modelo

- **Ensemble LightGBM + XGBoost con stacking:** Un meta-learner ligero (regresión logística) sobre las salidas de ambos modelos puede capturar errores sistemáticos que ninguno de los dos captura solo. El coste computacional en inferencia es despreciable (~2ms adicionales).
- **Modelo jerárquico para matchups escasos:** Para combinaciones bateador × lanzador con <10 PA históricos, el modelo actual regresa al promedio de la feature. Un modelo Bayesiano jerárquico (PyMC3 o Stan) con priors informados por los splits de grupo (ej. promedio de zurdos vs. lanzadores zurdos de velocidad media) reduciría el shrinkage hacia la media y mejoraría la precisión en matchups nuevos o rookies.

---

### 2.3 Explicabilidad y Confianza del Mánager

#### A. SHAP Values en el Scorecard

El `LineupScorecard` actual presenta "top 3 razones" en texto estático. La siguiente versión debería incluir:

- **SHAP waterfall por jugador:** Para cada bateador en el lineup recomendado, mostrar las 3 features con mayor contribución positiva y negativa a su E[R] individual. Ejemplo: "+0.08 E[R] por `xwoba_7d` elevado; -0.04 E[R] por `k_rate_7d` contra LHP".
- **Análisis contrafactual:** "Si mueves a [Jugador X] del slot 1 al slot 4, el E[R] cae 0.24 (de 4.82 a 4.58)". Este tipo de análisis "what-if" transforma el scorecard de un oráculo a una herramienta de exploración.

#### B. Dashboard de Comparación Histórica

El sistema registra todas las optimizaciones. Una vista de "decisiones similares pasadas" (misma mano del lanzador rival, park factor similar, condiciones meteorológicas comparables) con resultados reales daría al mánager el anclaje empírico que la intuición busca. Esto es complementario al experimento A/B ya implementado.

#### C. Modo Simulación Interactivo

Permitir que el mánager construya su propia alineación desde la UI y obtenga el E[R] calculado en tiempo real (fast_mode, ~3s), comparado contra la recomendación del sistema. Esta "prueba de concepto rápida" convierte la interacción de "imponer una recomendación" a "colaborar en la exploración".

---

## PARTE 3: Ruta hacia la Automatización Total y UI Final

### 3.1 Pasos de Automatización — Estado Actual

Los cuatro sprints de automatización identificados en la versión anterior están **completamente implementados**. A continuación el estado actual de cada uno.

#### Sprint A: Automatización de la Promoción del Modelo ✅ IMPLEMENTADO

**Módulo:** `src/mlops/auto_promoter.py` — `AutoPromoter`

El gate automático implementado evalúa dos condiciones (configurables vía variables de entorno):

```
Production_Candidate registrado con alias "staging" en MLflow
         ↓
Gate 1: Δlog-loss = champion_log_loss - challenger_log_loss ≥ PROMOTER_DELTA_LOG_LOSS (0.01)
Gate 2: challenger_ece_overall ≤ PROMOTER_ECE_CEILING (0.035)
         ↓
Si ambos gates pasan:
    - MLflow: set_registered_model_alias(name, "production", challenger_version)
    - MLflow: set_model_version_tag(old_champion, "status", "retired")
    - ECS: register_task_definition(...) + update_service(forceNewDeployment=True)
         ↓
Si algún gate falla → PromotionResult(promoted=False) — no se toca producción
```

En primera instalación (sin campeón en producción), el challenger se promueve directamente sin evaluación de delta. Cualquier fallo de MLflow o ECS lanza `PromotionError` para que el Airflow task quede marcado como fallido.

**Tests:** 17 tests en `tests/mlops/test_auto_promoter.py` — todos verdes.

#### Sprint B: Monitor de Disponibilidad de Jugadores ✅ IMPLEMENTADO

**Módulo:** `src/mlops/mlb_stats_monitor.py` — `MLBStatsMonitor`

El monitoreo continuo implementado cubre:

```
[APScheduler — cada 15 minutos]
         ↓
MLBStatsAPIClient.get_roster(team_id)     → /teams/{id}/roster/40Man
MLBStatsAPIClient.get_transactions(date)  → /transactions
         ↓
parse_roster() → status mapping:
    "Active" → "active"
    "10-Day IL" → "10day_il"
    "60-Day IL" → "60day_il"
    "Restricted List" → "restricted"
         ↓
PlayerAvailabilityStore.get_changed_player_ids()
    → detecta cambios comparando contra el estado almacenado en PostgreSQL
         ↓
PlayerAvailabilityStore.upsert_batch()
    → psycopg2 execute_values con ON CONFLICT DO UPDATE
         ↓
CacheInvalidator.invalidate(changed_player_ids)
    → redis.publish("optimizer:cache:invalidate", json_payload)
```

Fallos en cualquier capa no detienen el scheduler (`_safe_poll` captura todas las excepciones y las registra como warnings estructurados).

**Tests:** 18 tests en `tests/mlops/test_mlb_stats_monitor.py` — todos verdes.

#### Sprint C: Ingesta Automática de Scouting ✅ IMPLEMENTADO

**Módulo:** `src/rag/lambda_scouting_trigger.py` — handler AWS Lambda

Convención de clave S3: `scouting/{team_iso}/{scout_type}/{player_id}_{name}_{date}.{ext}`

- Tipos de scout válidos: `pro_scout`, `medical`, `developmental`, `analytics`
- Extensiones válidas: `txt`, `pdf`
- Extracción de texto: UTF-8 con fallback automático a latin-1
- En batches de 1 registro: re-lanza la excepción (activa DLQ de SQS para reintentos)
- En batches múltiples: error aislado por registro — incrementa `failed` y continúa

**Tests:** 17 tests en `tests/rag/test_lambda_scouting_trigger.py` — todos verdes.

#### Sprint D: Cierre Automático del Experimento A/B ✅ IMPLEMENTADO

**Módulo:** `src/experiment/post_game_resolver.py` — `PostGameResolver`

Job nocturno de Airflow que opera sobre una ventana de fechas configurables:

```
Para cada game_date en [hoy - lookback_days, hoy]:
    1. Lee particiones Parquet del ExperimentStore filtrando status="open"
    2. Obtiene el schedule del día desde MLB Stats API
    3. Construye mapa {abbrev → gamePk} (home y away)
    4. Por cada game_id con observación abierta:
         - Parsea {date}-{away}-{home} con regex
         - Busca gamePk en el mapa
         - Descarga boxscore → runs del home team
         - experiment_store.record_outcome(game_id, home_runs)
    5. Devuelve ResolveSummary con resolved / errors / open_found por fecha
```

**Tests:** 15 tests en `tests/experiment/test_post_game_resolver.py` — todos verdes.

#### Reporte Ejecutivo Mensual ✅ IMPLEMENTADO (nuevo en esta versión)

**Módulo:** `src/reporting/automated_reporter.py` — `AutomatedReporter`

PDF de 4 páginas generado con Plotly + kaleido + pypdf, distribuido por AWS SES:

| Página | Contenido |
|--------|-----------|
| 1 — Portada | KPIs: total partidos, ventaja del mánager (carreras), interpretación en español |
| 2 — Tendencia de adopción | Bar chart: partidos por tipo de divergencia en los últimos N días |
| 3 — Distribución de tipos | Pie chart (donut): order_only / player_swap / full_override |
| 4 — Precisión E[R] | Indicator (gauge) con delta actual vs. media; texto de estado si sin datos |

Callable de Airflow: `generate_monthly_report(execution_date="2026-05-01", send_email=True)`.

**Tests:** 20 tests en `tests/reporting/test_automated_reporter.py` — todos verdes.

#### Registro Automático de Divergencias ✅ IMPLEMENTADO (nuevo en esta versión)

**Endpoint:** `POST /v1/lineup/confirm` en `app/main.py`

Flujo completo implementado (ver sección 1.1, Capa 5 para detalles de diseño).

**Tests:** 18 tests en `tests/experiment/test_auto_divergence.py` — todos verdes.

---

### 3.2 Interfaz de Usuario (UX/UI) — El iPad en el Dugout

La UI final debe operar en tres modos según el momento del partido. Los tres componentes esenciales son:

#### Componente 1: Panel Pre-Partido (T-90min antes del primer pitch)

**Propósito:** El mánager ve la recomendación completa y puede explorar alternativas antes de decidir.

**Pantalla principal (iPad, orientación landscape):**

```
┌─────────────────────────────────────────────────────────────────┐
│  [SEMÁFORO] CONFIANZA: ALTA (0.87)                              │
│  "Datos frescos · Ventaja de platoon clara · Lineup estable"    │
├────────────────────────┬────────────────────────────────────────┤
│  ALINEACIÓN ÓPTIMA     │  ANÁLISIS DE CONFIANZA                 │
│                        │                                        │
│  1. García, A.  (SS)   │  Por posición:                         │
│  2. Torres, G.  (2B)   │  ●●● 1-2-3: Alta     ●●○ 4-5: Media  │
│  3. Judge, A.   (RF)   │  ●●● 6-7-8-9: Alta                   │
│  4. Stanton, G. (DH)   │                                        │
│  5. Rizzo, A.   (1B)   │  Razones top:                          │
│  6. Volpe, A.   (SS)   │  • Judge: OBP máx vs. LHP hoy         │
│  7. Soto, J.    (LF)   │  • Soto en slot 7: platoon +0.15 wOBA │
│  8. Wells, A.   (C)    │  • Stanton: exit velocity 7d 103 mph  │
│  9. Carpenter (3B)     │                                        │
│                        │  E[R] estimado: 4.82 carreras          │
│  [VER ALTERNATIVAS]    │  Win%: 58.3%                           │
├────────────────────────┴────────────────────────────────────────┤
│  [FLAGS DEL BENCH COACH]  +Añadir flag    [CONFIRMAR ALINEACIÓN]│
└─────────────────────────────────────────────────────────────────┘
```

**Flujos de acción:**
- **"Añadir flag":** Formulario rápido (jugador → tipo → severidad → peso [-1, +1] → nota de voz convertida a texto) → llama a `POST /v1/feedback/game` → el sistema recalcula con overrides activos en ~3s.
- **"Ver alternativas":** Modo simulación interactivo — el mánager arrastra jugadores a distintos slots y ve el impacto en E[R] en tiempo real (fast_mode).
- **"Confirmar alineación":** Si diverge del sistema, solicita confirmación + captura automáticamente `POST /v1/experiment/record-divergence`.

#### Componente 2: Vista de Reporte de Defensa (pantalla emergente, solo cuando hay crisis)

Cuando el sistema detecta una recomendación contraintuitiva (ej. STAR_PLAYER_MAJOR_DROP), antes de que el mánager pregunte, aparece una notificación:

```
⚠️  RECOMENDACIÓN INUSUAL DETECTADA
"El sistema propone a Judge en el slot 5, no en el 3"

[LEER INFORME DE DEFENSA]     [IGNORAR Y CONTINUAR]
```

El informe RAG se renderiza como Markdown nativo en iPad, con 5 secciones: qué recomienda, los 3 datos cuantitativos, contexto de scouts, cuándo ignorar la recomendación y la alternativa canónica.

#### Componente 3: Review Post-Partido (T+30min tras el último out)

**Propósito:** Cerrar el ciclo de aprendizaje con el mínimo fricción posible.

```
┌─────────────────────────────────────────────────────────────────┐
│  REVISIÓN POST-PARTIDO  —  NYY 7, BOS 3                         │
├─────────────────────────────────────────────────────────────────┤
│  ¿Qué tan útil fue la recomendación AI hoy?                     │
│  ★★★★☆  (4/5)                                                   │
│                                                                  │
│  ¿Hubo contexto que el sistema no vio?                          │
│  "Judge tenía una contractura leve, por eso lo bajé al 5"       │
│                                                                  │
│  ¿La decisión real resultó correcta?  [Sí]  [No]               │
│                                                                  │
│  COMPARATIVA:  AI esperaba 4.82 C  |  Resultado real: 7 C       │
│  Tu alineación: +2.18 C vs expectativa del sistema              │
├─────────────────────────────────────────────────────────────────┤
│  [ENVIAR FEEDBACK]     ← 3 segundos de trabajo. Gracias.        │
└─────────────────────────────────────────────────────────────────┘
```

Esta pantalla llama automáticamente a `POST /v1/feedback/game` (post_game), `POST /v1/experiment/record-outcome` y recuerda al analista que registre si hubo flags de jugadores para el siguiente partido.

---

### 3.3 Cierre del Ciclo de Aprendizaje Continuo

El objetivo es que cada decisión del mánager, ya sea acordar con el sistema o divergir, contribuya automáticamente a mejorar el modelo del día siguiente. El flujo completo es:

```
[DUGOUT — iPad del Mánager]
         │
         ├─ Pre-partido: flags cualitativos del bench coach
         │    ↓ POST /v1/feedback/game (pre_game)
         │    ↓ FeedbackStore → data/gold/qualitative_feedback/
         │
         ├─ Divergencia de alineación capturada automáticamente
         │    ↓ POST /v1/lineup/confirm  ← nuevo — compara vs. caché AI
         │    ↓ ExperimentStore → data/gold/ab_experiment/
         │
         └─ Post-partido: evaluación + resultado real
              ↓ POST /v1/feedback/game (post_game)
              ↓ POST /v1/experiment/record-outcome  ← automatizado: PostGameResolver (Airflow)
              │
              ▼
[PIPELINE NOCTURNO — Airflow DAG]
         │
         ├─ Capa 2: FeedbackStore.get_sample_weights(last_7d)
         │    → sample_weights Parquet → LightGBM retraining
         │    → jugadores flaggeados con physical_illness ese día
         │       cuentan menos en el ajuste del modelo
         │
         ├─ Capa 3: FeedbackStore.analyze_recurring_flags(30d)
         │    → flags recurrentes → FeatureProposals
         │    → revisión semanal humana → si aprobada,
         │       nueva feature binaria integrada al pipeline de features
         │
         └─ Experimento A/B: ExperimentStore.get_summary(90d)
              → si manager_advantage > +0.5 C/partido
                 en tipo "full_override" con muestra ≥ 15:
                 → alerta al equipo ML: posible conocimiento táctico
                    no capturado en los features actuales
```

**Tiempo de ciclo completo:**
- Flag pre-partido → afecta optimización: **instantáneo** (en la misma solicitud)
- Flag post-partido → afecta reentrenamiento: **24–48 horas** (siguiente ciclo nocturno)
- Flag recurrente → nueva feature en el modelo: **2–3 semanas** (revisión humana + pipeline de features)
- Divergencia A/B → evidencia estadística: **30–60 días** (mínimo 15 observaciones cerradas para significancia)

---

## Apéndice: Mapa de Archivos del Sistema

```
MLB AI/
├── app/
│   ├── main.py              # FastAPI service — todos los endpoints
│   │                        #   + _CachedRecommendation, _AppState.lineup_cache
│   │                        #   + POST /v1/lineup/confirm (auto-divergence)
│   └── schemas.py           # Pydantic v2 contracts
│                            #   + LineupConfirmRequest / LineupConfirmResponse
│
├── src/
│   ├── ingestion/           # Statcast → Bronze Delta
│   ├── quality/             # Great Expectations data quality
│   ├── weather/             # Datos meteorológicos del estadio
│   ├── features/            # Rolling features + platoon splits
│   ├── embeddings/          # Player embeddings (Pinecone)
│   ├── models/              # LightGBM AtBatPredictor
│   ├── simulation/          # Motor Monte Carlo (Numba JIT)
│   ├── optimizer/           # Genetic Algorithm lineup optimizer
│   ├── rag/
│   │   ├── lambda_scouting_trigger.py  # ★ S3 Lambda → Pinecone ingestion
│   │   └── ...              # RAG + Claude tactical briefings
│   ├── mlops/
│   │   ├── auto_promoter.py            # ★ Champion/challenger gate + ECS deploy
│   │   ├── mlb_stats_monitor.py        # ★ Roster/injury poll → PostgreSQL + Redis
│   │   └── ...              # MLflow registry + retraining pipeline
│   ├── trust/               # LineupScorecard + confidence scoring
│   ├── crisis/              # Crisis detection + RAG defense reports
│   ├── feedback/            # Qualitative overrides + FeedbackStore
│   ├── experiment/
│   │   ├── post_game_resolver.py       # ★ Airflow nightly A/B outcome closure
│   │   └── ...              # ExperimentStore, classify_divergence
│   └── reporting/
│       └── automated_reporter.py       # ★ Monthly PDF report + AWS SES
│
├── tests/
│   ├── conftest.py
│   ├── experiment/
│   │   ├── test_api.py                 # 19 tests
│   │   ├── test_auto_divergence.py     # ★ 18 tests — /v1/lineup/confirm
│   │   ├── test_classify.py            # 8 tests
│   │   ├── test_interpretation.py      # 12 tests
│   │   ├── test_post_game_resolver.py  # ★ 15 tests
│   │   └── test_store.py              # 17 tests
│   ├── mlops/
│   │   ├── test_auto_promoter.py       # ★ 17 tests
│   │   └── test_mlb_stats_monitor.py   # ★ 18 tests
│   ├── rag/
│   │   └── test_lambda_scouting_trigger.py  # ★ 17 tests
│   └── reporting/
│       └── test_automated_reporter.py  # ★ 20 tests
│                            # TOTAL: 161 tests — todos verdes
│
├── data/
│   ├── bronze/              # Raw Statcast Delta Lake
│   ├── silver/              # Cleaned plate appearances
│   └── gold/
│       ├── batter_features_rolling/  # Feature matrix
│       ├── qualitative_feedback/     # FeedbackStore Parquet
│       └── ab_experiment/            # ExperimentStore Parquet
│
├── models/
│   └── at_bat_predictor.pkl          # Modelo campeón en producción
│
└── mlruns/                           # MLflow experiments + model registry
```

★ = módulo añadido en esta versión

---

## Estado de los Entregables del Plan de Adopción del Staff

| Entregable | Estado | Endpoint(s) / Módulo(s) |
|------------|--------|-------------------------|
| Protocolo A/B Testing | ✅ Implementado — 71 tests | `POST /v1/experiment/record-divergence` · `POST /v1/experiment/record-outcome` · `GET /v1/experiment/summary` |
| Registro automático de divergencias | ✅ **Nuevo** — 18 tests | `POST /v1/lineup/confirm` |
| Dashboard de Confianza (LineupScorecard) | ✅ Implementado | Campo `scorecard` en `POST /v1/optimize/lineup` |
| Sistema de Retroalimentación Cualitativa | ✅ Implementado (3 capas) | `POST /v1/feedback/game` · `GET /v1/feedback/overrides` · `GET /v1/feedback/feature-proposals` |
| Protocolo de Crisis y Fallback | ✅ Implementado | Campo `defense_report` en `POST /v1/optimize/lineup` |
| Promotor automático de modelos | ✅ **Nuevo** — 17 tests | `src/mlops/auto_promoter.py` · Airflow PythonOperator |
| Monitor de disponibilidad de jugadores | ✅ **Nuevo** — 18 tests | `src/mlops/mlb_stats_monitor.py` · APScheduler 15min |
| Ingesta automática de informes de scouts | ✅ **Nuevo** — 17 tests | `src/rag/lambda_scouting_trigger.py` · AWS Lambda S3 |
| Cierre automático del experimento A/B | ✅ **Nuevo** — 15 tests | `src/experiment/post_game_resolver.py` · Airflow DAG |
| Reporte ejecutivo mensual PDF | ✅ **Nuevo** — 20 tests | `src/reporting/automated_reporter.py` · AWS SES |
| UI iPad para el dugout | Pendiente — Parte 3 de esta hoja de ruta | N/A |

**Tests totales: 161 — todos verdes.** (56 protocolo A/B original → 143 tras automatización → 161 tras registro automático de divergencias)

---

---

## Historial de Versiones

| Versión | Fecha | Cambios principales |
|---------|-------|---------------------|
| 1.0 | Mayo 2026 | Auditoría inicial. Protocolo A/B Testing, LineupScorecard, Retroalimentación Cualitativa, Crisis/Fallback. 56 tests. |
| 2.0 | Mayo 2026 | **Bloque de automatización MLOps completo**: AutoPromoter, MLBStatsMonitor, lambda_scouting_trigger, PostGameResolver, AutomatedReporter (87 tests nuevos, 143 total). **Registro automático de divergencias**: `POST /v1/lineup/confirm` con cache en `_AppState`, idempotencia y TTL 24h (18 tests nuevos, 161 total). 6 operaciones manuales automatizadas (33% del total). |

---

*Documento generado a partir de auditoría técnica completa del repositorio. Las estimaciones de latencia se basan en benchmarks locales (CPU: 4 cores / 16GB RAM). Los tiempos de producción variarán según la infraestructura de despliegue (AWS EC2 c5.4xlarge recomendado para optimización full-mode con latencia ≤ 30s).*
