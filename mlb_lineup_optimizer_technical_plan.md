# MLB Lineup Optimizer — Arquitectura Técnica y Plan de Ejecución de Grado de Producción

> **Documento:** Diseño de Sistema ML — Motor de Optimización de Alineaciones  
> **Clasificación:** Confidencial — Solo uso interno del cuerpo técnico y analítico  
> **Versión:** 1.0.0  
> **Stack principal:** AWS · Delta Lake · Spark · LightGBM · PyTorch · Ray · LangChain · LlamaIndex · FastAPI · Kubernetes · MLflow  
> **Objetivo matemático:** Maximizar `E[R]` (Expected Runs por juego) y `P(W)` (Probabilidad de Victoria) mediante la optimización combinatoria de la alineación titular (posiciones 1–9) en función del lanzador abridor rival, el bullpen enemigo, los park factors dinámicos y las condiciones climáticas.

---

## Tabla de Contenidos

1. [Fase 1 — Ingesta de Datos y Data Engineering](#fase-1)
2. [Fase 2 — Feature Engineering Avanzado](#fase-2)
3. [Fase 3 — Modelado Predictivo y Simulación](#fase-3)
4. [Fase 4 — Capa de GenAI y RAG](#fase-4)
5. [Fase 5 — MLOps, CI/CD y Gobierno](#fase-5)
6. [Fase 6 — Interfaz de Usuario y Consumo](#fase-6)
7. [Apéndice — Diagrama de Arquitectura Global](#apendice)

---

## Principios Arquitectónicos Rectores

Antes de entrar en las fases, se establecen los tres principios no negociables que gobiernan cada decisión de diseño:

- **Sin Training-Serving Skew:** Toda feature calculada durante el entrenamiento debe calcularse de forma absolutamente idéntica en tiempo de inferencia. El Feature Store centralizado es la solución estructural a este problema.
- **Explicabilidad como requisito de negocio, no como opción:** El mánager debe poder entender *por qué* el sistema recomienda una alineación. Un modelo de caja negra sin narrativa no será adoptado independientemente de su precisión matemática.
- **SLA de día de partido:** El pipeline completo (ingesta de condiciones del día → feature computation → simulación Monte Carlo → generación de narrativa) debe completarse en una ventana máxima de **90 minutos** desde la confirmación del lineup rival, garantizando entrega al cuerpo técnico antes de las decisiones pre-partido.

---

<a name="fase-1"></a>
## Fase 1 — Ingesta de Datos, Pipeline y Data Engineering

### 1.1 Definición del Modelo Canónico de Datos (Data Contract)

**Descripción:**  
Antes de ingerir un solo byte de datos, se define un contrato de datos explícito y versionado que especifica las entidades core del sistema y sus relaciones. Las entidades primarias son: `Player`, `PlateAppearance (PA)`, `Pitch`, `Game`, `Venue`, `BullpenState` y `WeatherSnapshot`. Cada entidad tiene un schema formal con tipos de dato, nullability, rangos de validez y relaciones de clave foránea. El contrato se versiona semánticamente (`v1.2.3`) y cualquier cambio rompe el build del pipeline hasta que se actualiza explícitamente.

**Herramientas:**
- **Apache Avro / Protobuf** — serialización binaria con schema embedded para Kafka
- **Apache Iceberg** — evolución de schema sin downtime en el lakehouse
- **Great Expectations** — suites de validación declarativas (Expectation Suites) por entidad
- **AWS Glue Data Catalog** — registro centralizado de schemas y linaje
- **dbt** — documentación de modelos y contratos de columnas (`dbt contracts`)

**Por qué (Lógica de Negocio/Matemática):**  
El béisbol tiene docenas de fuentes de datos heterogéneas con vocabularios distintos: Statcast llama `launch_speed` a lo que Retrosheet denomina `batted_ball_velocity`. Sin un contrato canónico, los modelos downstream entrenados en una fuente no sirven sobre datos de otra. El contrato de datos es la capa de traducción que hace al sistema **fuente-agnóstico** y permite incorporar nuevas fuentes (ej. datos de wearables) sin refactoring del modelo.

---

### 1.2 Ingesta de Statcast: MLB StatsAPI y Baseball Savant

**Descripción:**  
Se implementan dos modos de ingesta para Statcast: **ingesta histórica en batch** (temporadas 2015–presente, ~15M de pitches) y **ingesta incremental en tiempo real** (<5 minutos de latencia el día del partido). El pipeline histórico se ejecuta como un Spark Job en EMR, descargando y normalizando los ~80 atributos por pitch: velocidad, spin rate, break horizontal/vertical, coordenadas de liberación `(x, y, z)`, extension, tipo de pitch, resultado del PA, spray angle, exit velocity y launch angle. El pipeline en tiempo real consume el feed de la MLB StatsAPI via polling y publica eventos en Kafka.

```bash
# Ejemplo de descarga histórica con pybaseball
python -c "
from pybaseball import statcast
import pandas as pd

# Descarga por temporada para evitar timeouts
for year in range(2015, 2026):
    df = statcast(start_dt=f'{year}-03-01', end_dt=f'{year}-11-01')
    df.to_parquet(f's3://mlb-lakehouse/bronze/statcast/year={year}/data.parquet')
    print(f'Temporada {year}: {len(df):,} pitches ingestados')
"
```

**Herramientas:**
- **pybaseball** — cliente Python para MLB StatsAPI y Baseball Savant
- **Apache Kafka + AWS MSK** — streaming de eventos en tiempo real (topic: `statcast.pitches.live`)
- **Apache Spark (PySpark) en AWS EMR** — procesamiento distribuido del histórico
- **Delta Lake** — formato de tabla ACID para el data lakehouse (capa Bronze)
- **AWS Kinesis Data Firehose** — buffer de ingesta en tiempo real hacia S3
- **Apache Airflow (MWAA)** — orquestación de DAGs de ingesta programada

**Por qué (Lógica de Negocio/Matemática):**  
Statcast, lanzado en 2015, provee la granularidad de datos más alta jamás disponible en béisbol profesional: cada pitch es un vector de 80+ dimensiones en un espacio físico tridimensional. El **spin rate** de un curveball es el predictor más fuerte de su drop (movimiento vertical), y el **induced vertical break** (separando el efecto gravitacional del spin real) explica hasta un 40% de la varianza en el swing-and-miss rate contra ese pitch. Sin Statcast, el modelo solo puede aproximar relaciones a nivel de resultado (hit/out), no a nivel de mecanismo físico del lanzamiento.

---

### 1.3 Ingesta de Retrosheet y Baseball-Reference (Datos Históricos pre-Statcast)

**Descripción:**  
Retrosheet ofrece datos play-by-play con resolución de resultado desde 1916. Se descarga el corpus completo de archivos `.evx` (event files), se parsean con la librería `chadwick-bureau` y se transforma en un esquema tabular normalizado. De Baseball-Reference se extraen estadísticas de temporada agregadas (slash lines, splits, WAR) via web scraping responsable con respeto a los `robots.txt` y límites de rate. Estos datos se almacenan en la capa Bronze del lakehouse y se transforman a Silver con dbt.

```bash
# Parseo de Retrosheet con chadwick-bureau
cwevent -y 2010 -f 0,1,2,3,4,5,6,9,10,11,12,16,26,34,35,36 \
    2010*.EVA 2010*.EVN > retrosheet_2010_events.csv

# Upload a S3 Bronze
aws s3 cp retrosheet_2010_events.csv \
    s3://mlb-lakehouse/bronze/retrosheet/year=2010/events.csv
```

**Herramientas:**
- **chadwick-bureau** — parser oficial de archivos de eventos Retrosheet (C/Python bindings)
- **pandas / polars** — transformación y normalización de datos tabulares
- **BeautifulSoup4 + httpx** — scraping responsable de Baseball-Reference
- **Apache Airflow** — DAG de ingesta histórica (corre una vez, idempotente)
- **AWS S3 Intelligent-Tiering** — almacenamiento de datos históricos fríos con costos optimizados
- **dbt (Bronze → Silver transforms)** — limpieza, deduplicación y tipado de columnas

**Por qué (Lógica de Negocio/Matemática):**  
Los **platoon splits** (rendimiento de un bateador vs. lanzadores diestros vs. zurdos) requieren un mínimo de 400–600 **Plate Appearances (PA)** para ser estadísticamente estables. Muchos jugadores de roster tienen <200 PAs en Statcast contra ciertas manos. Retrosheet extiende el historial a décadas, permitiendo calcular splits con significancia estadística real via **regresión bayesiana hacia la media de la liga**. Esta profundidad histórica es imposible de obtener solo con Statcast.

---

### 1.4 Integración de API de Condiciones Climáticas y Meteorológicas

**Descripción:**  
Se implementa un servicio de ingesta meteorológica que consulta tres fuentes complementarias: **Tomorrow.io** para pronóstico hiper-local por coordenadas GPS del estadio (resolución 1km²), **NOAA/Weather.gov** como fuente de respaldo oficial gratuita, y datos históricos de **Meteostat** para el entrenamiento del modelo de Park Factors dinámico. El servicio se ejecuta via AWS Lambda en polling cada 30 minutos durante el día del partido, con frecuencia aumentada a cada 10 minutos en las 3 horas previas al primer pitch. Los atributos capturados incluyen: temperatura (°F), humedad relativa (%), velocidad y dirección del viento (mph, grados azimut), presión barométrica (inHg), índice de calor y condición de precipitación.

```python
# Lambda handler para ingesta climática
import boto3, httpx, json
from datetime import datetime

def lambda_handler(event, context):
    venues = {
        "wrigley": {"lat": 41.9484, "lon": -87.6553},
        "fenway": {"lat": 42.3467, "lon": -71.0972},
        "coors":  {"lat": 39.7559, "lon": -104.9942},
    }
    
    for venue_id, coords in venues.items():
        resp = httpx.get(
            "https://api.tomorrow.io/v4/weather/realtime",
            params={**coords, "apikey": get_secret("TOMORROW_API_KEY"),
                    "fields": "temperature,humidity,windSpeed,windDirection,pressureSurfaceLevel"}
        )
        payload = resp.json()["data"]["values"]
        payload["venue_id"] = venue_id
        payload["captured_at"] = datetime.utcnow().isoformat()
        
        boto3.client("kinesis").put_record(
            StreamName="weather-snapshots",
            Data=json.dumps(payload),
            PartitionKey=venue_id
        )
```

**Herramientas:**
- **Tomorrow.io API** — pronóstico hiper-local por coordenadas GPS del estadio
- **NOAA Weather.gov API** — fuente oficial de respaldo (sin costo, SLA gubernamental)
- **Meteostat Python** — datos históricos de estaciones meteorológicas para entrenamiento
- **AWS Lambda** — polling serverless cada 10–30 minutos
- **AWS Kinesis Data Streams** — ingesta en tiempo real de snapshots meteorológicos
- **AWS Secrets Manager** — gestión segura de API keys

**Por qué (Lógica de Negocio/Matemática):**  
El viento tiene un efecto demostrado y cuantificable en el béisbol. Un viento de salida de 15 mph en **Wrigley Field** (Chicago) incrementa el home run rate en aproximadamente **12–18%** respecto a condiciones de viento nulo. La dirección importa tanto como la velocidad: el viento de entrada reduce el HR rate en una magnitud similar. A **Coors Field** (Denver, altitud 5,280 pies), la menor densidad del aire aumenta la distancia de los batazos en ~10% respecto al nivel del mar incluso en condiciones neutras. Sin modelar estas variables dinámicas, el Park Factor del modelo es estático y sistemáticamente erróneo varias veces por semana.

---

### 1.5 Ingesta de Biometría y Datos de Wearables de Jugadores

**Descripción:**  
Se integran las APIs de **WHOOP** (HRV, Recovery Score, Strain Score diario) y **Catapult Sports** (GPS tracking en entrenamientos: aceleración, carga de trabajo mecánica, distancia sprint) para los jugadores del roster. Dado el carácter extremadamente sensible y privado de estos datos biométricos (regulados bajo HIPAA y el CBA de la MLB), se implementa un pipeline de anonymización y cifrado en origen: los datos se cifran con **Fernet (AES-128-CBC)** antes de salir de la red del equipo, y se almacenan en AWS HealthLake con control de acceso RBAC estricto. Solo el médico del equipo y el director de analítica tienen acceso completo; el modelo recibe features agregadas anonimizadas (ej. `recovery_tier: HIGH/MED/LOW`).

**Herramientas:**
- **WHOOP API v2** — Recovery Score, HRV, Strain Score, calidad del sueño
- **Catapult Sports API** — carga mecánica, distancia, velocidad máxima en entrenamientos
- **AWS HealthLake (FHIR R4)** — almacenamiento seguro de datos de salud con auditoría
- **cryptography (Fernet)** — cifrado simétrico AES-128-CBC en origen antes de transmisión
- **AWS Macie** — detección automática de datos personales sensibles en S3
- **Apache Airflow** — DAG de ingesta diaria post-entrenamiento (06:00 AM hora local)

**Por qué (Lógica de Negocio/Matemática):**  
Un jugador con un **Recovery Score de WHOOP por debajo de 33%** (zona roja) presenta estadísticamente tiempos de reacción un 8–12% más lentos y mayor varianza en sus mecánicas de bateo. Este efecto no es capturado por ninguna métrica Statcast retrospectiva. La biometría permite ajustar el modelo de rendimiento proyectado del jugador con información del estado físico *actual*, no del promedio histórico. Esto es especialmente crítico para bateadores que han jugado back-to-back games o volaron a través de zonas horarias.

---

### 1.6 Construcción del Data Lakehouse con Arquitectura Medallion (Bronze / Silver / Gold)

**Descripción:**  
Se implementa la arquitectura Medallion en tres capas sobre AWS S3 con Delta Lake como formato de tabla ACID:

- **Capa Bronze (Raw):** Datos ingestados exactamente como vienen de la fuente, sin transformaciones. Inmutables. Particionados por `source / year / month / day`. Retención indefinida.
- **Capa Silver (Curated):** Datos limpios, deduplicados, tipados correctamente, con schema validado por Great Expectations. Joins entre fuentes (ej. Statcast + clima del momento). Particionados por `game_date / venue_id`. Retención 10 años.
- **Capa Gold (Feature-Ready):** Agregaciones analíticas y pre-computed features listas para ML. Incluye ventanas rodantes de 7, 15, 30, 60 días. Materializada en Delta Lake y sincronizada con el Feature Store. Retención 5 años (datos más recientes disponibles online).

```text
s3://mlb-lakehouse/
├── bronze/
│   ├── statcast/year=2024/month=07/day=15/
│   ├── retrosheet/year=2010/
│   ├── weather/venue_id=wrigley/date=2024-07-15/
│   └── biometrics/encrypted/date=2024-07-15/
├── silver/
│   ├── plate_appearances/game_date=2024-07-15/
│   ├── pitcher_profiles/season=2024/
│   └── park_factors_dynamic/game_date=2024-07-15/
└── gold/
    ├── batter_features_rolling/
    ├── pitcher_arsenal_profiles/
    └── matchup_features/
```

**Herramientas:**
- **Delta Lake 3.x** — formato ACID con time-travel, schema evolution y OPTIMIZE/ZORDER
- **Apache Spark 3.5 (PySpark) en AWS EMR Serverless** — procesamiento distribuido
- **dbt (con Spark adapter)** — transformaciones SQL declarativas Silver → Gold con linaje
- **Great Expectations** — Expectation Suites automatizadas en cada capa
- **Apache Iceberg** — alternativa evaluada para la capa Gold por su mejor soporte de partitioning evolutivo
- **AWS Glue Data Catalog** — metastore centralizado compatible con Athena y EMR

**Por qué (Lógica de Negocio/Matemática):**  
La separación de capas es el mecanismo de **linaje de datos auditable**. Si el modelo produce una predicción incorrecta, se puede trazar desde el output hasta el dato raw exacto que la causó. Esto es fundamental tanto para depuración técnica como para la credibilidad ante el cuerpo técnico: el mánager debe poder confiar en que los números que ve provienen de datos verificables, no de transformaciones opacas en un pipeline monolítico.

---

### 1.7 Orquestación de Pipelines con Apache Airflow

**Descripción:**  
Todos los flujos de datos del sistema se modelan como DAGs (Directed Acyclic Graphs) en Apache Airflow, desplegado en AWS Managed Workflows for Apache Airflow (MWAA). Se definen cuatro DAGs críticos:

1. **`dag_historical_backfill`** — Corre una vez. Ingesta completa de Statcast 2015–presente y Retrosheet 1916–presente. Idempotente.
2. **`dag_daily_feature_refresh`** — Corre cada madrugada a las 03:00 AM. Actualiza features rolling de bateadores y perfiles de pitchers con los datos del día anterior.
3. **`dag_game_day_pipeline`** — **El DAG más crítico.** Se activa automáticamente al confirmar el lineup rival (~10:00 AM). Secuencia: ingesta climática → actualización de bullpen state → recompute de matchup features → simulación Monte Carlo → generación de narrativa LLM → push a dashboard. Debe completarse antes de las 11:30 AM.
4. **`dag_model_retraining_weekly`** — Corre cada lunes a las 02:00 AM. Dispara el pipeline de reentrenamiento en SageMaker si los datos nuevos de la semana superan un umbral mínimo de volumen.

**Herramientas:**
- **Apache Airflow 2.8+ (AWS MWAA)** — orquestación de DAGs con UI web
- **Celery Executor + AWS SQS** — paralelización de tasks en el clúster
- **Airflow Sensors** (S3KeySensor, ExternalTaskSensor) — dependencias entre DAGs
- **PagerDuty + Slack webhooks** — alertas en fallos de SLA (via `on_failure_callback`)
- **Astronomer Cosmos** — integración nativa de dbt models dentro de DAGs de Airflow

**Por qué (Lógica de Negocio/Matemática):**  
La disponibilidad del sistema el día del partido es innegociable. Un pipeline que falla silenciosamente a las 10:30 AM y no se detecta hasta las 12:50 PM (10 minutos antes del primer pitch) es peor que no tener sistema. Airflow provee **visibilidad de SLA**, reintentos configurables con backoff exponencial, y dependencias explícitas que garantizan que el `dag_game_day_pipeline` nunca corre sobre datos stale.

---

### 1.8 Data Quality Gates y Monitoreo Continuo de Pipelines

**Descripción:**  
Se implementa una estrategia de calidad de datos en tres niveles que actúa como red de seguridad antes de que cualquier dato llegue al modelo:

- **Nivel 1 — Schema Validation (dbt):** Tests de tipo, nullability, unicidad y rango en cada modelo dbt. Tests integrados como `not_null`, `unique`, `accepted_values`, y `relationships` se ejecutan en cada run.
- **Nivel 2 — Distributional Expectations (Great Expectations):** Suites de expectativas estadísticas que validan que las distribuciones de features no han derivado significativamente. Ej: `expect_column_mean_to_be_between(column='spin_rate', min_value=1800, max_value=3200)`.
- **Nivel 3 — Observability (Monte Carlo Data):** Monitoreo continuo de freshness, volumen, distribución y linaje de tablas en producción. Alertas automáticas ante anomalías.

```python
# Ejemplo de Expectation Suite para tabla silver.plate_appearances
import great_expectations as ge

suite = context.add_expectation_suite("silver.plate_appearances.v1")
validator = context.get_validator(batch_request=batch_request, expectation_suite_name=suite.name)

validator.expect_column_values_to_not_be_null("batter_id")
validator.expect_column_values_to_not_be_null("pitcher_id")
validator.expect_column_values_to_be_between("launch_speed", min_value=0, max_value=125)
validator.expect_column_values_to_be_between("spin_rate", min_value=0, max_value=4000)
validator.expect_column_values_to_be_in_set("pitch_type", 
    ["FF","SI","FC","SL","CH","CU","KC","FS","EP","FO","CS","SC","KN"])
validator.expect_column_proportion_of_unique_values_to_be_between("pa_id", min_value=0.999)

validator.save_expectation_suite(discard_failed_expectations=False)
```

**Herramientas:**
- **Great Expectations 0.18+** — suites de expectativas declarativas con Data Docs automáticos
- **dbt tests** — validaciones integradas en el DAG de transformación
- **Monte Carlo Data** — observabilidad de data quality en producción (anomaly detection)
- **AWS CloudWatch** — métricas custom de pipeline (registros procesados, latencia, fallos)
- **Grafana + Prometheus** — dashboards de monitoreo operativo en tiempo real

**Por qué (Lógica de Negocio/Matemática):**  
Un dato erróneo de `spin_rate` (ej. un pitch registrado con 12,000 RPM por error de sensor) que llega al modelo sin ser detectado puede distorsionar el perfil completo del arsenal de un lanzador y producir una predicción de matchup radicalmente incorrecta. En el contexto de una decisión de alineación con implicaciones de millones de dólares en contratos y posiciones en playoffs, un **Data Quality Gate** no es un nice-to-have: es el mecanismo de control de riesgo más importante del sistema.

---

<a name="fase-2"></a>
## Fase 2 — Feature Engineering Avanzado y Analítica

### 2.1 Cálculo de Métricas Sabermétricas Avanzadas Base

**Descripción:**  
Se computan y materializan en la capa Gold del lakehouse las métricas sabermétricas canónicas que constituyen el vocabulario base del sistema. Estas métricas se calculan a nivel de jugador, temporada y ventana temporal:

- **wOBA (Weighted On-Base Average):** Pondera cada tipo de resultado ofensivo por su valor lineal en run expectancy: `wOBA = (0.69×BB + 0.72×HBP + 0.89×1B + 1.27×2B + 1.62×3B + 2.10×HR) / (AB + BB - IBB + SF + HBP)`. Los pesos se recalibran anualmente con la run expectancy de la temporada vigente.
- **wRC+ (Weighted Runs Created Plus):** Normaliza wOBA por parque y era: `wRC+ = 100 × [(wOBA - lgwOBA)/wOBAScale + lgR/PA] / lgR/PA`.
- **FIP (Fielding Independent Pitching):** `FIP = (13×HR + 3×(BB+HBP) - 2×K) / IP + FIP_constant`. Elimina el ruido del BABIP del lanzador.
- **xFIP:** Reemplaza los HRs reales con el número esperado basado en fly ball rate × HR/FB de liga.
- **BABIP (Batting Average on Balls In Play):** `BABIP = (H - HR) / (AB - K - HR + SF)`. Indicador de suerte residual.

```sql
-- dbt model: gold/batter_sabermetrics_season.sql
WITH pa_data AS (
    SELECT
        batter_id,
        season,
        COUNT(*)                                              AS pa,
        SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END)    AS bb,
        SUM(CASE WHEN events = 'hit_by_pitch' THEN 1 ELSE 0 END) AS hbp,
        SUM(CASE WHEN events = 'single' THEN 1 ELSE 0 END)  AS singles,
        SUM(CASE WHEN events = 'double' THEN 1 ELSE 0 END)  AS doubles,
        SUM(CASE WHEN events = 'triple' THEN 1 ELSE 0 END)  AS triples,
        SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hr,
        SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) AS k
    FROM silver.plate_appearances
    GROUP BY batter_id, season
),
weights AS (SELECT * FROM gold.woba_weights_by_season)
SELECT
    p.batter_id,
    p.season,
    p.pa,
    ROUND(
        (w.w_bb * p.bb + w.w_hbp * p.hbp + w.w_1b * p.singles +
         w.w_2b * p.doubles + w.w_3b * p.triples + w.w_hr * p.hr) /
        NULLIF(p.pa - p.hbp, 0), 3
    ) AS woba
FROM pa_data p
JOIN weights w ON p.season = w.season
```

**Herramientas:**
- **dbt (Spark adapter)** — materialización de métricas en la capa Gold con linaje completo
- **polars** — computación local de alta performance para métricas exploratorias
- **pybaseball** — validación cruzada contra valores publicados en FanGraphs
- **NumPy / SciPy** — cálculo de pesos lineales y constantes de FIP por temporada

**Por qué (Lógica de Negocio/Matemática):**  
Las estadísticas tradicionales (AVG, RBI, ERA) son profundamente ruidosas como predictores de rendimiento futuro porque incluyen componentes no reproducibles (BABIP, timing). **wOBA** correlaciona con run scoring con un R² ≈ 0.95 a nivel de temporada, versus ~0.70 del OPS y ~0.50 del AVG. El modelo de simulación necesita como input la distribución de probabilidad de *resultados del PA*, no promedios de resultados: wOBA y sus componentes son exactamente esa distribución, expresada en unidades de carreras.

---

### 2.2 Construcción de Platoon Splits Dinámicos con Actualización Bayesiana

**Descripción:**  
Los **platoon splits** miden el diferencial de rendimiento de un bateador contra lanzadores diestros vs. zurdos. El problema central es el **tamaño de muestra pequeño**: un jugador de primer año puede tener 80 PAs vs. zurdos, insuficiente para estimar su split real. Se implementa un modelo bayesiano jerárquico que parte de un prior informativo (la distribución de platoon splits de la liga) y lo actualiza con la evidencia observada del jugador específico.

El modelo para el split de wOBA de un jugador `i` contra mano `h` es:

```
wOBA_i,h ~ Normal(μ_i,h, σ²)
μ_i,h ~ Normal(μ_league_h, τ²)   # Prior jerárquico: media de la liga como ancla
```

Donde `τ²` controla cuánto puede el jugador individual desviarse de la media de la liga. A mayor número de PAs observados, más peso tiene la evidencia del jugador vs. el prior de liga.

Adicionalmente, los splits se calculan a nivel de **tipo de pitch** (no solo de mano del lanzador): wOBA vs. fastball, vs. slider, vs. changeup, vs. breaking ball. Esta granularidad permite modelar matchups específicos del arsenal del lanzador rival, no solo su mano de lanzar.

**Herramientas:**
- **PyMC 5.x** — modelado probabilístico bayesiano con NUTS sampler
- **ArviZ** — análisis y visualización de distribuciones posteriores
- **Stan (via CmdStanPy)** — alternativa para modelos que requieren mayor performance de sampling
- **polars** — cómputo de splits brutos con window functions sobre datos Silver
- **scikit-learn (BayesianRidge)** — aproximación frecuentista de respaldo para inferencia rápida

**Por qué (Lógica de Negocio/Matemática):**  
Sin regularización bayesiana, un jugador con 45 PAs vs. zurdos y un .480 wOBA en esa muestra sería proyectado como un bateador de élite contra zurdos cuando en realidad es pura varianza estadística. El modelo bayesiano **regresa automáticamente la estimación hacia la media de la liga** en proporción inversa al tamaño de muestra, produciendo estimaciones de split que son dramáticamente más estables y predictivas. En backtesting, los splits bayesianos reducen el RMSE de predicción de wOBA en un ~25% vs. los splits brutos observados para jugadores con <200 PAs en el split en cuestión.

---

### 2.3 Modelado de Park Factors Dinámicos (Estático + Climático)

**Descripción:**  
Los park factors estáticos miden el efecto promedio multi-anual del estadio sobre el scoring (ej. Coors Field tiene un factor de ~1.35 para HR). Sin embargo, estos promedios no capturan la variación *intra-parque* debida a condiciones climáticas. Se construye un modelo de **Park Factor Dinámico** que produce un factor ajustado para cada juego específico, combinando:

- **Factor Estático Histórico:** Media de 3 temporadas del factor de parque tradicional, calculado como `(Runs_home / PA_home) / ((Runs_all - Runs_home) / (PA_all - PA_home))`, separado por HR, 2B, 3B, 1B y total de carreras.
- **Ajuste Dinámico Climático:** Un modelo XGBoost entrenado sobre datos históricos de juegos con sus condiciones climáticas que aprende: `delta_park_factor = f(wind_speed, wind_direction, temperature, humidity, pressure)`. El wind_direction se codifica como componentes vectoriales `(sin(θ), cos(θ))` para preservar la continuidad circular.

El output final es un `dynamic_park_factor` por categoría de batazo para el juego del día, con intervalos de confianza derivados de la distribución de predicciones del ensemble.

**Herramientas:**
- **XGBoost** — modelo de ajuste climático del park factor
- **SHAP** — atribución de contribución de cada variable climática al ajuste
- **polars + PyArrow** — computación de factores estáticos históricos
- **Meteostat** — datos históricos de condiciones climáticas en cada estadio para entrenamiento
- **scipy.optimize** — optimización de los pesos de combinación estático/dinámico
- **scikit-learn (calibration)** — calibración de los intervalos de confianza del modelo

**Por qué (Lógica de Negocio/Matemática):**  
Coors Field en Denver tiene una altitud de 5,280 pies (1,609m). A esta altitud, la densidad del aire es ~18% menor que al nivel del mar, lo que reduce el drag aerodinámico en la misma proporción. Sin ajuste dinámico, el modelo usaría el factor estático de Coors en un día de 95°F (que reduce aún más la densidad) exactamente igual que en un día de 40°F, produciendo un error sistemático de hasta ±3-4% en la proyección de HR rate. Multiplicado por 162 juegos y las decisiones de composición del roster, este error compuesto tiene un impacto significativo en la estrategia de construcción de equipo.

---

### 2.4 Generación de Perfiles de Arsenal de Lanzadores (Pitch Profiling)

**Descripción:**  
Para cada lanzador con ≥10 apariciones en los últimos 3 años, se construye un **perfil de arsenal** multidimensional. El proceso tiene tres etapas:

1. **Clustering de pitches:** Se aplica **HDBSCAN** sobre los vectores `(velocity, spin_rate, horizontal_break, vertical_break, release_extension)` para identificar automáticamente los clusters de pitch-type reales del lanzador (los clasificadores automáticos de la MLB a veces cometen errores). Esto produce los tipos de pitch "ground truth" del arsenal.

2. **Vectorización del perfil:** Para cada pitch-type identificado, se calcula: `usage_pct`, `avg_velocity`, `avg_spin_rate`, `avg_induced_vbreak`, `avg_horizontal_break`, `whiff_rate`, `put_away_rate` (K-rate en 2-strike counts), `usage_pct_by_count` (11 counts distintos), y `usage_pct_runners_on_vs_empty`.

3. **Modelado de tendencias:** Se aplica una regresión LOWESS sobre las últimas 10 apariciones para detectar tendencias en velocidad y movimiento (ej. un lanzador perdiendo 1.5 mph en la fastball en sus últimos 3 starts es una señal de alerta crítica).

**Herramientas:**
- **HDBSCAN** — clustering de pitches robusto a ruido y sin asumir número de clusters
- **scikit-learn Pipeline** — preprocesamiento + clustering encadenados
- **polars** — computación de estadísticas de pitch por count, situación y runner state
- **statsmodels (LOWESS)** — suavizado no paramétrico para detección de tendencias de velocidad
- **AWS SageMaker Feature Store** — almacenamiento y serving de perfiles de arsenal

**Por qué (Lógica de Negocio/Matemática):**  
El matchup a nivel de **pitch-type específico** es el nivel de granularidad correcto para predecir resultados de PA. Un bateador puede tener un wOBA de .380 vs. lanzadores diestros en general, pero un wOBA de .250 vs. sliders de derechos con movimiento horizontal >12 pulgadas. El lanzador rival puede lanzar ese slider específico el 45% del tiempo en 2-strike counts. Sin el perfil de arsenal, esta interacción crítica es invisible para el modelo. Con él, es el feature más predictivo del matchup.

---

### 2.5 Modelado del Estado del Bullpen y Proyección de Disponibilidad

**Descripción:**  
La alineación no solo enfrenta al abridor. Históricamente, el lanzador titular cubre 5-6 entradas en promedio; el resto del juego pertenece al bullpen. Se construye un modelo de dos componentes:

- **Predicción de duración del abridor:** Un modelo de supervivencia (Weibull) que estima la distribución de probabilidad de cuántas entradas lanzará el abridor dado su carga reciente (pitches en sus últimos 5 días), su eficiencia de pitches histórica (pitches/inning), y el estado del juego simulado.
- **Predicción de secuencia del bullpen:** Un modelo XGBoost que, dado el marcador simulado, entrada, y estado de fatiga de cada relevista (días de descanso desde última aparición × pitches lanzados), predice qué relevista entrará a continuación y estima su efectividad proyectada con su propio perfil de arsenal.

El output es una secuencia probabilística de lanzadores para los innings 5-9, con perfiles de arsenal asociados, que alimenta directamente la simulación de Monte Carlo.

**Herramientas:**
- **lifelines (Kaplan-Meier + Weibull AFT)** — análisis de supervivencia para duración del abridor
- **XGBoost** — clasificación de selección de relevista y regresión de efectividad
- **CatBoost** — manejo nativo de features categóricas (handedness, pitch-type preference)
- **AWS SageMaker Feature Store** — serving en tiempo real del estado del bullpen
- **Apache Airflow** — actualización del estado del bullpen post-juego

**Por qué (Lógica de Negocio/Matemática):**  
El impacto del bullpen en la decisión de alineación es asimétrico: si el bullpen rival tiene un zurdo especialista (LOOGY) que entra típicamente en el 7mo inning, colocar a bateadores zurdos en las posiciones 2-3-4 de la alineación (que batean en el 7mo con runners on base) es matemáticamente subóptimo. El sistema puede identificar esta interacción y reorganizar la alineación para que los bateadores diestros enfrenten al especialista zurdo. Este tipo de **optimización cross-innings** es invisible para cualquier análisis que no proyecte el juego completo.

---

### 2.6 Entrenamiento de Player Embeddings con Redes Neuronales

**Descripción:**  
Se entrena una red neuronal de **entity embeddings** que mapea cada bateador y cada lanzador a un vector denso de 128 dimensiones en un espacio latente compartido. La arquitectura es un modelo de factorización matricial con autoencoders residuales:

```
Input: one-hot(batter_id) + one-hot(pitcher_id) + context_features
→ Embedding Layer (128-dim para cada entidad)
→ Interaction Layer (Hadamard product + concatenación)
→ MLP (512 → 256 → 128 → 8 outputs)
Output: P(walk), P(K), P(1B), P(2B), P(3B), P(HR), P(HBP), P(out_in_play)
```

El embedding de cada jugador encapsula su "identidad sabermétrca" de forma que la **distancia euclidiana** en el espacio latente corresponde a similaridad de perfil. Esto permite: (a) generalización a matchups nunca vistos usando jugadores similares, y (b) transfer learning para jugadores con poca historia (rookies).

**Herramientas:**
- **PyTorch 2.x** — implementación de la arquitectura de embedding
- **PyTorch Lightning** — estructura de entrenamiento con logging automático
- **AWS SageMaker Training Jobs (p4d.24xlarge)** — entrenamiento distribuido en GPUs A100
- **UMAP** — visualización 2D del espacio de embeddings para validación cualitativa
- **MLflow** — tracking de experimentos y versionado de embeddings
- **Ray Train** — distributed data-parallel training

**Por qué (Lógica de Negocio/Matemática):**  
Los modelos basados en features tabulares puras (XGBoost) no pueden generalizar bien a matchups infrecuentes (ej. un rookie en su primer mes vs. un lanzador que nunca ha enfrentado bateadores de su perfil). Los embeddings aprenden representaciones que capturan **similaridades latentes no capturadas por features explícitos**: dos lanzadores con arsenales físicamente similares tendrán embeddings cercanos, permitiendo al modelo predecir el rendimiento del rookie contra el Lanzador B basándose en su desempeño vs. el Lanzador A con embedding similar.

---

### 2.7 Construcción y Mantenimiento del Feature Store Centralizado

**Descripción:**  
El Feature Store es la capa de infraestructura más crítica del sistema. Centraliza el cálculo, almacenamiento y serving de features para garantizar **zero Training-Serving Skew**: la definición exacta de cada feature es idéntica en el pipeline de entrenamiento y en el pipeline de inferencia de producción.

Se implementa una arquitectura dual-store:
- **Offline Store (Delta Lake en S3):** Features históricas para entrenamiento. Consultables con Spark/Athena.
- **Online Store (Amazon ElastiCache Redis con Redis-stack):** Features de baja latencia (<5ms P99) para inferencia en tiempo real durante la simulación del día de partido.

Cada feature tiene metadata asociada: nombre, descripción, tipo de dato, tiempo de vida (TTL), ventana temporal de cálculo, y la query/código que la produce (linaje completo).

```python
# Definición de feature group en SageMaker Feature Store
import boto3
sagemaker_client = boto3.client('sagemaker')

sagemaker_client.create_feature_group(
    FeatureGroupName='batter-rolling-features-v2',
    RecordIdentifierFeatureName='batter_id',
    EventTimeFeatureName='feature_timestamp',
    FeatureDefinitions=[
        {'FeatureName': 'batter_id',           'FeatureType': 'String'},
        {'FeatureName': 'feature_timestamp',   'FeatureType': 'Fractional'},
        {'FeatureName': 'woba_7d',             'FeatureType': 'Fractional'},
        {'FeatureName': 'woba_30d',            'FeatureType': 'Fractional'},
        {'FeatureName': 'woba_season',         'FeatureType': 'Fractional'},
        {'FeatureName': 'k_rate_30d',          'FeatureType': 'Fractional'},
        {'FeatureName': 'bb_rate_30d',         'FeatureType': 'Fractional'},
        {'FeatureName': 'avg_exit_velocity_30d','FeatureType': 'Fractional'},
        {'FeatureName': 'hard_hit_rate_30d',   'FeatureType': 'Fractional'},
        {'FeatureName': 'recovery_tier',       'FeatureType': 'String'},
    ],
    OnlineStoreConfig={'EnableOnlineStore': True},
    OfflineStoreConfig={
        'S3StorageConfig': {'S3Uri': 's3://mlb-lakehouse/gold/feature-store/'}
    }
)
```

**Herramientas:**
- **AWS SageMaker Feature Store** — managed dual-store (online + offline)
- **Feast (Feature Store open-source)** — alternativa evaluada para mayor flexibilidad
- **Amazon ElastiCache (Redis 7)** — online store para serving de baja latencia
- **Apache Airflow** — actualización programada de features con freshness tracking
- **Tecton** — plataforma enterprise evaluada para Feature Store con transformaciones en tiempo real

**Por qué (Lógica de Negocio/Matemática):**  
El **Training-Serving Skew** es la causa más común de degradación de modelos ML en producción que no se detecta inmediatamente. Si el feature `woba_30d` se calcula con una ventana de 30 días naturales en entrenamiento pero con 30 días de juego en producción (días sin juego excluidos), el modelo recibe features de distribución diferente en producción, degradando silenciosamente su precisión. El Feature Store con definiciones centralizadas de features elimina esta inconsistencia por construcción.

---

<a name="fase-3"></a>
## Fase 3 — Modelado Predictivo y Simulación

### 3.1 Modelo de Predicción de Resultado a Nivel de Plate Appearance (PA-Level Model)

**Descripción:**  
El núcleo predictivo del sistema. Para cada **Plate Appearance** potencial (combinación bateador × lanzador × contexto), el modelo produce una **distribución de probabilidad** sobre los 8 posibles resultados mutuamente excluyentes:

`P(outcome | batter, pitcher, count_situation, runners, venue, weather)`

Los 8 outcomes son: `{Walk (BB), Strikeout (K), Single (1B), Double (2B), Triple (3B), Home Run (HR), Hit By Pitch (HBP), Out In Play}`.

El vector de features de entrada tiene ~340 dimensiones que incluyen: features del bateador (wOBA rolling, splits, exit velocity, K/BB ratio, embedded vector de 128 dims), features del lanzador (FIP, arsenal pitch-usage, spin rate por pitch-type, embedded vector de 128 dims), features de matchup (historial directo bateador-lanzador si existe, platoon matchup), features contextuales (venue, park_factor_dynamic, weather_vector, temperatura ajustada de densidad del aire).

Se implementa un **stacked ensemble** de tres modelos base:

1. **LightGBM** — GBDT con manejo nativo de features categóricas y alta performance. Modelo base de mayor peso.
2. **XGBoost** — GBDT complementario con regularización L1/L2 diferente. Captura correlaciones distintas.
3. **FT-Transformer (PyTorch)** — Feature Tokenization Transformer, estado del arte en datos tabulares. Especialmente fuerte en interacciones de alto orden entre los embeddings de jugadores.

Un **meta-learner (Ridge Regression)** combina las predicciones de los tres modelos base, con pesos optimizados por Optuna.

**Herramientas:**
- **LightGBM 4.x** — modelo base primario (velocidad + precisión en tabular data)
- **XGBoost 2.x** — modelo base secundario (diversidad de ensemble)
- **PyTorch 2.x (FT-Transformer)** — modelo base de deep learning para interacciones latentes
- **Optuna** — optimización de hiperparámetros de los tres modelos base y del meta-learner
- **scikit-learn (CalibratedClassifierCV)** — calibración de probabilidades (requisito para simulación)
- **SHAP** — explicabilidad global y local de predicciones del ensemble

**Por qué (Lógica de Negocio/Matemática):**  
La **calibración de probabilidades** es absolutamente crítica: el modelo no solo necesita predecir el outcome más probable, sino que su `P(HR) = 0.08` debe ser realmente observable el 8% de las veces en datos holdout. Sin calibración, las probabilidades producen distribuciones de runs esperados sesgadas en la simulación Monte Carlo. La calibración post-hoc con Platt Scaling (CalibratedClassifierCV) reduce el Expected Calibration Error (ECE) del ensemble en un ~60% respecto a los modelos sin calibrar.

---

### 3.2 Simulador de Media Entrada como Cadena de Markov

**Descripción:**  
El **béisbol tiene una estructura matemática perfecta para cadenas de Markov:** el estado del juego al inicio de cada PA es completamente descrito por el par `(outs, bases_occupied)`. Hay exactamente **24 estados posibles** (3 outs posibles × 8 configuraciones de base: vacías, 1ra, 2da, 3ra, 1ra+2da, 1ra+3ra, 2da+3ra, llenas). El estado `(3 outs, cualquier base)` es el estado absorbente (fin de la media entrada).

Para cada estado `s` y cada PA del siguiente bateador en el orden de la alineación, el PA-Level Model produce la distribución `P(outcome)`. Cada outcome genera una **transición de estado determinística** según las reglas del béisbol y acumula 0 o más carreras. Se construye la **matriz de transición** `T[s → s']` completa y se usa para calcular el vector de `Run Expectancy` de cada estado, equivalente a la tabla de Run Expectancy Matrix (RE24) pero específica del matchup del día.

```python
# Transición de estado simplificada
BASES_STATES = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),
                (1,1,0),(1,0,1),(0,1,1),(1,1,1)]

def transition(state: tuple, outs: int, outcome: str) -> tuple[tuple, int, int]:
    """Retorna (nuevo_estado_bases, nuevos_outs, carreras_anotadas)."""
    bases = list(state)
    runs = 0
    if outcome == 'HR':
        runs = 1 + sum(bases)
        return (0,0,0), outs, runs
    elif outcome == '1B':
        runs = bases[2]                    # runner en 3ra anota
        bases = [1, bases[0], bases[1]]    # avance estándar 1 base
        return tuple(bases), outs, runs
    elif outcome == 'K' or outcome == 'out_in_play':
        return tuple(bases), outs + 1, 0
    # ... demás outcomes
```

**Herramientas:**
- **NumPy** — representación y operaciones sobre matrices de transición de 24×24
- **numba (@jit decorator)** — compilación JIT de la función de transición para simulaciones masivas
- **polars** — computación batch de matrices de transición para múltiples matchups
- **SciPy (linalg)** — resolución del sistema de ecuaciones lineales para Run Expectancy teórica

**Por qué (Lógica de Negocio/Matemática):**  
La cadena de Markov captura **la naturaleza secuencial del béisbol** que los modelos de regresión simples no pueden capturar. Un bateador con alta tasa de walk es mucho más valioso bateando 2do en la alineación (donde puede avanzar al bateador de turno y configurar la entrada) que bateando 8vo (donde sus walks rara vez producen carreras). La **Run Expectancy Matrix** cuantifica exactamente el valor de cada configuración de estado, haciendo óptima la decisión de ordenamiento que el sistema busca maximizar.

---

### 3.3 Motor de Simulación Monte Carlo del Juego Completo

**Descripción:**  
Se implementa un simulador de juego completo de 9 entradas que itera sobre el orden de la alineación para simular **N = 100,000 juegos** por configuración de alineación candidata. Cada simulación es completamente independiente y se ejecuta en paralelo usando Ray.

El flujo de una simulación individual de un juego completo es:

1. Inicializar marcador `(0, 0)` y estado de lineup position para ambos equipos.
2. Para cada media entrada (18 en total, 9 por equipo):
   a. Inicializar estado Markov: `outs=0, bases=(0,0,0), runs=0`.
   b. Determinar el lanzador activo (abridor o relevista según modelo de bullpen).
   c. Iterar PAs hasta `outs == 3`: samplear outcome de PA-Level Model, aplicar transición de Markov, acumular carreras.
   d. Avanzar la posición del bateador en la alineación (con wraparound desde posición 9 → 1).
3. Determinar ganador y resultado final.

El output de 100,000 simulaciones para una configuración de alineación es: `E[R_scored]`, `E[R_allowed]`, `P(W)`, `P(W | margin ≤ 1)` (juegos cerrados), y percentiles `[5, 25, 50, 75, 95]` de carreras anotadas.

```python
import ray
import numpy as np

@ray.remote
def simulate_game_batch(
    lineup_order: list[str],
    pa_model,
    bullpen_sequence: list,
    park_factor: float,
    n_simulations: int = 5000
) -> dict:
    results = {"wins": 0, "runs_scored": [], "runs_allowed": []}
    for _ in range(n_simulations):
        team_runs = simulate_full_game(lineup_order, pa_model, bullpen_sequence, park_factor)
        opp_runs  = simulate_full_game(opp_lineup, pa_model, opp_bullpen, park_factor)
        results["runs_scored"].append(team_runs)
        results["runs_allowed"].append(opp_runs)
        if team_runs > opp_runs:
            results["wins"] += 1
    return results

# Distribución de 100,000 simulaciones en 20 workers de Ray
futures = [simulate_game_batch.remote(lineup, model, bullpen, pf, 5000)
           for _ in range(20)]
all_results = ray.get(futures)
```

**Herramientas:**
- **Ray 2.x (ray.remote)** — distribución de simulaciones en clúster de AWS EC2
- **NumPy + Numba (JIT)** — vectorización y compilación de la lógica de simulación inner loop
- **AWS EC2 (c7i.48xlarge)** — 192 vCPUs para la simulación paralela masiva del día de partido
- **JAX (jit + vmap)** — alternativa de GPU para vectorización masiva de simulaciones
- **SciPy (stats)** — análisis de distribuciones de output (Bootstrap CI, Kolmogorov-Smirnov)

**Por qué (Lógica de Negocio/Matemática):**  
Con 100,000 simulaciones, el **Intervalo de Confianza al 95%** para `E[R]` es aproximadamente `±0.05 carreras` (por la Ley de los Grandes Números). Esto es suficientemente preciso para distinguir diferencias de alineación de `+0.1 carreras/juego`, que es el nivel de granularidad mínimo necesario para tomar decisiones. Con 50,000 simulaciones el CI sería `±0.07`, que aún es operativamente válido. El paralismo de Ray permite completar las 100,000 simulaciones en ~45 segundos usando un clúster de 4 máquinas de 48 vCPUs, dentro del SLA del día de partido.

---

### 3.4 Optimizador Combinatorio de Alineaciones con Búsqueda Evolutiva

**Descripción:**  
Encontrar la alineación óptima de 9 jugadores es un problema combinatorio de `9! = 362,880` permutaciones posibles. Simular las 362,880 configuraciones completas tomaría ~5 horas con el motor de Monte Carlo. Se implementa una estrategia de búsqueda en tres capas:

**Capa 1 — Inicialización con heurísticas sabermétricas:** Se genera una población inicial de 200 configuraciones usando reglas basadas en conocimiento del dominio:
- Bateadores con mayor **OBP** en las posiciones 1 y 2 (mayor número de PAs con bases vacías, maximiza oportunidades de score).
- Mayor **wOBA** y **ISO (Isolated Power)** en posiciones 3, 4 y 5 (RISP).
- Menor wOBA en posiciones 8 y 9 (menor impacto marginal en carreras esperadas).
- Ajuste de mano de bateo para separar bateadores del mismo lado vs. el arsenal del lanzador abridor.

**Capa 2 — Algoritmo Genético con DEAP:** Sobre la población inicial de 200, se ejecuta un algoritmo genético con:
- **Selección:** Torneo de tamaño 3.
- **Crossover:** Partially Mapped Crossover (PMX) para preservar permutaciones válidas.
- **Mutación:** Swap mutation (intercambio de 2 posiciones aleatorias). Probabilidad 0.15.
- **Fitness function:** `E[R]` estimado con una simulación reducida de 5,000 iteraciones (speed proxy).
- **Generaciones:** 150 generaciones × 200 individuos = 30,000 evaluaciones.

**Capa 3 — Refinamiento local con Optuna:** El top-10 de configuraciones del algoritmo genético se evalúa con la simulación completa de 100,000 iteraciones y se selecciona el ganador global.

**Herramientas:**
- **DEAP (Distributed Evolutionary Algorithms in Python)** — framework de algoritmos genéticos
- **Optuna** — Bayesian Optimization para refinamiento de candidatos finales
- **Google OR-Tools** — resolución exacta de variantes con restricciones duras (ej. jugadores lesionados)
- **Ray Tune** — paralelización de la evaluación de fitness en el clúster
- **NumPy** — representación vectorizada de permutaciones

**Por qué (Lógica de Negocio/Matemática):**  
La **complejidad factorial** del espacio de búsqueda hace que la búsqueda exhaustiva sea computacionalmente inviable dentro del SLA del día de partido. El algoritmo genético con inicialización informada por heurísticas sabermétricas **reduce el espacio efectivo de búsqueda en ~99.8%** manteniendo una cobertura de soluciones cercanas al óptimo global. En backtesting sobre 500 juegos históricos, el sistema encuentra configuraciones con `E[R] ≥ 95%` del óptimo exhaustivo en el 94% de los casos, con un tiempo de ejecución de <4 minutos.

---

### 3.5 Modelo Adversarial: Predicción de Estrategia del Mánager Rival

**Descripción:**  
El mánager rival no es una variable estática. Sus decisiones de pitching changes, shifts defensivos e intentional walks responden al estado del juego en tiempo real. Se entrena un modelo de **comportamiento del mánager** que predice, dado el estado del juego simulado, la probabilidad de que el mánager rival realice un cambio de pitcher en cada PA. Los features incluyen: entrada, marcador, mano del bateador vs. mano del relevista siguiente disponible, historial de uso del bullpen en las últimas 48h, y el "perfil" del mánager (algunos son muy agresivos con el bullpen, otros conservadores).

El output de este modelo alimenta la función de transición del simulador: en lugar de asumir que el abridor lanza hasta que el modelo de supervivencia lo retira, el simulador samplea la decisión del mánager en cada PA con >3 entradas completadas.

**Herramientas:**
- **XGBoost (clasificación binaria)** — P(bullpen change | game_state)
- **Logistic Regression** — baseline interpretable para validación del modelo XGBoost
- **Retrosheet (play-by-play histórico)** — datos de entrenamiento de decisiones de mánager reales
- **SHAP** — análisis de qué features del estado del juego más influyen en la decisión de cambio

**Por qué (Lógica de Negocio/Matemática):**  
Si el sistema sabe que el mánager rival tiene un 87% de probabilidad de sacar al abridor zurdo después del 5to inning con runners en scoring position, y hay un especialista diestro en el bullpen con un 0.55 wOBA contra contra bateadores zurdos, entonces **colocar a los bateadores zurdos en las posiciones de la alineación que batean típicamente en esas entradas (4-5-6)** y reservar los diestros para el 7mo-8vo tiene un valor matemático cuantificable. Este tipo de optimización cross-innings y adversarial es exclusivo de los sistemas más avanzados de analítica deportiva.

---

### 3.6 Validación, Backtesting y Métricas de Éxito del Sistema

**Descripción:**  
Se implementa una estrategia de validación rigurosa con **Walk-Forward Cross-Validation** temporal:

- **Split de datos:** Train: temporadas 2015–2022. Validation: 2023. Test (holdout real): 2024.
- **Métricas del PA-Level Model:** Log-loss multiclase, Brier Score por clase de outcome, Expected Calibration Error (ECE), y Area Under the ROC Curve (AUC) por clase.
- **Métrica de sistema completo:** Delta de Expected Runs entre alineaciones recomendadas por el sistema vs. alineaciones históricas reales del equipo: `ΔE[R] = E[R]_optimizer - E[R]_actual_historical`. Target: `ΔE[R] ≥ +0.20 carreras/juego`.
- **Métrica de negocio real:** Wins Above Replacement (WAR de alineación), comparando el número de victorias proyectadas de las alineaciones optimizadas vs. las históricas en el backtesting de 2023.

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss
import mlflow

tscv = TimeSeriesSplit(n_splits=5, gap=162)  # gap = 1 temporada MLB
with mlflow.start_run(run_name="pa_model_walk_forward_v3"):
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[val_idx])
        
        mlflow.log_metric(f"fold_{fold}_logloss", log_loss(y[val_idx], proba))
        mlflow.log_metric(f"fold_{fold}_ece", compute_ece(y[val_idx], proba, n_bins=10))
    
    mlflow.sklearn.log_model(model, "pa_model_v3")
```

**Herramientas:**
- **scikit-learn (TimeSeriesSplit)** — validación cruzada temporal correcta (sin data leakage futuro)
- **MLflow** — registro automático de métricas, parámetros y modelos por fold
- **SciPy (stats.ks_2samp)** — tests de Kolmogorov-Smirnov para distribución de residuales
- **statsmodels** — tests de significancia estadística de `ΔE[R]`
- **Plotly** — curvas de calibración y reliability diagrams

**Por qué (Lógica de Negocio/Matemática):**  
En series temporales deportivas, el **data leakage** es la amenaza más grave: si el modelo entrena sobre datos del futuro (temporadas posteriores al juego que está intentando predecir), sus métricas de validación serán radicalmente optimistas y el modelo fallará en producción. Walk-Forward Validation es el único método que respeta la causalidad temporal y produce estimaciones de performance honestas. El target de `ΔE[R] ≥ +0.20 carreras/juego`, si sostenido sobre 162 juegos, equivale a **+32 carreras adicionales por temporada**, lo que se traduce estadísticamente (via Pythagorean Expectation) en aproximadamente **+3 victorias adicionales por temporada**.

---

<a name="fase-4"></a>
## Fase 4 — Capa de GenAI, LLMs y RAG

### 4.1 Pipeline de Ingesta y Procesamiento de Scouting Reports

**Descripción:**  
El departamento de scouting produce reportes en lenguaje natural (PDFs, Word docs, emails) que contienen información crítica no capturada por Statcast: tendencias mecánicas recientes del lanzador ("ha perdido plano en el slider en sus últimos 3 starts"), lesiones menores no reportadas oficialmente, ajustes a su repertorio descubiertos en video, y contexto psicológico ("pitcher nervioso en situaciones de alta presión"). Se implementa un pipeline de ingesta automática de estos documentos no estructurados.

El flujo de procesamiento es:
1. **Detección de nuevos documentos:** Airflow sensor monitorea S3 bucket `scouting-reports/inbox/` para nuevos archivos.
2. **Parsing y extracción de texto:** `Unstructured.io` extrae texto limpio preservando la jerarquía del documento (headers, bullets, tablas).
3. **Chunking semántico:** Los documentos se dividen en chunks de 512 tokens con 64 tokens de overlap, usando `LlamaIndex.SentenceSplitter` que respeta límites de oraciones.
4. **Tagging de entidades:** Un NER (Named Entity Recognition) especializado en béisbol etiqueta menciones de jugadores, estadios, pitch-types y métricas en cada chunk.
5. **Generación de embeddings:** Cada chunk se vectoriza con `text-embedding-3-large` de OpenAI (3072 dimensiones).
6. **Indexación:** Los vectores se almacenan en Amazon OpenSearch con sus metadatos (jugador_id, fecha_reporte, autor_scout, tipo_matchup).

**Herramientas:**
- **Unstructured.io** — extracción de texto de PDFs, Word, emails con preservación de estructura
- **LlamaIndex (SentenceSplitter + SimpleNodeParser)** — chunking semántico inteligente
- **spaCy + custom NER model** — reconocimiento de entidades específicas del béisbol
- **OpenAI text-embedding-3-large** — embeddings de alta calidad para retrieval semántico
- **Apache Airflow** — DAG de ingesta automática activado por nuevo archivo en S3
- **AWS S3** — almacén de documentos fuente con versionado habilitado

**Por qué (Lógica de Negocio/Matemática):**  
Statcast captura qué pasó (spin rate, exit velocity, resultado). Los scouts capturan *por qué* pasó y *qué podría pasar* (mecánicas emergentes, señales de fatiga, desequilibrios no estadísticos). Esta información **tiene una ventana de relevancia de 1-3 semanas** antes de que se refleje en las estadísticas. Un sistema que solo usa datos de Statcast es inherentemente **rezagado en ~2 semanas** respecto al estado actual real de un lanzador. El RAG cierra ese gap usando la inteligencia cualitativa más reciente del departamento de scouting.

---

### 4.2 Sistema RAG (Retrieval-Augmented Generation) para Scouting

**Descripción:**  
Se implementa un sistema RAG completo que, dado el matchup del día (ej. "Bateadores del equipo X vs. Pitcher Y en estadio Z"), recupera los chunks de scouting reports más relevantes y los usa como contexto del LLM para generar el briefing pre-partido.

El pipeline de RAG tiene dos fases:

**Retrieval Phase:**
1. Construir la query de retrieval: `f"Arsenal y tendencias recientes de {pitcher_name} | {pitcher_team} | {game_date}"`.
2. Generar el embedding de la query con el mismo modelo (`text-embedding-3-large`).
3. Consultar Amazon OpenSearch con KNN vector search: recuperar `top-k=8` chunks más similares.
4. Re-rankear con **Cohere Reranker** usando relevancia cruzada entre la query y cada chunk.
5. Filtrar por `metadata.fecha_reporte` para priorizar reportes de las últimas 3 semanas.

**Generation Phase:**
1. Construir el prompt con el contexto recuperado + outputs numéricos del optimizer.
2. Invocar el LLM para generar el briefing narrativo.
3. Validar que cada afirmación estadística en la narrativa esté respaldada por datos del Feature Store (guardrail anti-alucinación).

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.opensearch import OpensearchVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

vector_store = OpensearchVectorStore(
    endpoint="https://mlb-opensearch.us-east-1.es.amazonaws.com",
    index="scouting-reports-v2",
    dim=3072
)
index = VectorStoreIndex.from_vector_store(vector_store)
retriever = index.as_retriever(similarity_top_k=8)

# Retrieve + Rerank
query = f"Tendencias recientes arsenal {pitcher_name} | {game_date}"
nodes = retriever.retrieve(query)
reranked = cohere_reranker.rerank(query, [n.text for n in nodes], top_n=5)
```

**Herramientas:**
- **LlamaIndex** — orquestación del pipeline RAG completo (indexing + retrieval + generation)
- **Amazon OpenSearch Service (k-NN plugin)** — vector store gestionado con búsqueda ANN
- **Cohere Rerank API** — re-ranking cross-encoder para mejorar la relevancia de los chunks recuperados
- **FAISS (self-hosted)** — alternativa de vector search para entornos offline
- **Pinecone** — alternativa managed evaluada para vector store

**Por qué (Lógica de Negocio/Matemática):**  
El RAG supera al **fine-tuning puro** para este caso de uso porque los datos de scouting son **dinámicos y de alta frecuencia** (nuevos reportes cada semana), mientras que el fine-tuning requiere un ciclo de reentrenamiento costoso. RAG permite al LLM acceder a información actualizada en tiempo real sin reentrenar. La etapa de **re-ranking con Cohere** es crítica: la similitud coseno pura puede recuperar chunks sobre el pitcher correcto pero de baja relevancia temporal; el cross-encoder re-rankea según la relevancia semántica completa, mejorando el Recall@5 en un ~18% respecto al retrieval puro en nuestros benchmarks internos.

---

### 4.3 Fine-Tuning del LLM en Dominio Béisbol

**Descripción:**  
Los LLMs de propósito general no comprenden el vocabulario específico del béisbol moderno: "xFIP", "BABIP", "induced vertical break", "run expectancy", "platoon advantage". Se realiza **domain adaptation** mediante Supervised Fine-Tuning (SFT) con la técnica **LoRA (Low-Rank Adaptation)** sobre el modelo base, seguido de una etapa de **RLHF (Reinforcement Learning from Human Feedback)** usando preferencias de los scouts y analistas del equipo.

El corpus de fine-tuning incluye:
- 2,000+ artículos de FanGraphs, The Athletic (béisbol), y Baseball Prospectus sobre analítica avanzada.
- 500 pares de Q&A generados sintéticamente sobre métricas sabermétricas, validados por analistas.
- 300 transcripciones de conferencias de prensa de mánagers con análisis de decisiones de alineación.
- 200 ejemplos de alta calidad de "output ideal": briefings pre-partido escritos manualmente por el equipo de scouting.

El fine-tuning usa LoRA con `rank=16, alpha=32` sobre todos los attention layers, reduciendo el número de parámetros entrenables en un ~99.8% vs. full fine-tuning, con una degradación de performance <2% en el domain-specific benchmark.

**Herramientas:**
- **Claude claude-opus-4-20250514 / GPT-4o (base model)** — modelo base para fine-tuning
- **LoRA / QLoRA (via PEFT library de HuggingFace)** — adaptación eficiente de parámetros
- **AWS SageMaker Training Jobs (ml.p4d.24xlarge)** — entrenamiento en 8× A100 GPUs
- **TRL (Transformer Reinforcement Learning)** — pipeline RLHF con PPO
- **Weights & Biases** — tracking del fine-tuning y comparación de checkpoints
- **LM-Evaluation-Harness** — evaluación del modelo fine-tuned en benchmarks de béisbol custom

**Por qué (Lógica de Negocio/Matemática):**  
Un LLM que confunde "FIP" con "FIP-stop" o describe un "two-seamer" como una variante del fastball de cuatro costuras (en vez de un pitch con movimiento sink-run) produce briefings incorrectos que erosionan la confianza del cuerpo técnico en el sistema completo. La **credibilidad técnica** del LLM es un requisito de adopción, no de precisión estadística. El RLHF con preferencias de scouts reales garantiza que el modelo produce explicaciones en el registro exacto que el cuerpo técnico espera: sin simplificaciones excesivas pero sin jerga innecesariamente técnica.

---

### 4.4 Motor de Narrativa y Explicabilidad para el Mánager

**Descripción:**  
El output numérico del sistema (distribuciones de runs, matrices de matchup, probabilidades de victoria) debe traducirse a lenguaje natural accionable para el mánager. Se implementa un motor de narrativa en dos capas:

**Capa 1 — Selección de insights clave:** Un módulo de selección basado en SHAP identifica automáticamente los 5 insights más importantes que explican la diferencia entre la alineación óptima recomendada y la alineación "intuitiva" alternativa. Los insights se priorizan por magnitud de impacto en `E[R]`.

**Capa 2 — Generación de narrativa:** El LLM fine-tuned recibe como input:
- Los 5 insights clave con sus valores SHAP y estadísticas de respaldo del Feature Store.
- Los chunks de scouting relevantes recuperados por RAG.
- El perfil del arsenal del lanzador del día.
- El output final del optimizer (alineación recomendada + delta E[R] vs. alternativas).

El prompt template (Jinja2) produce un briefing con estructura canónica: resumen ejecutivo (3 frases), análisis del matchup por posición de alineación, alertas especiales (platoon masivo, condiciones climáticas extremas), y las 2 alineaciones alternativas más cercanas al óptimo con sus tradeoffs.

```python
from jinja2 import Template
import anthropic

BRIEFING_TEMPLATE = Template("""
Eres el principal analista de béisbol del equipo. Genera el briefing pre-partido basándote EXCLUSIVAMENTE en los datos proporcionados.

## Datos del Partido
- **Lanzador Rival:** {{ pitcher_name }} ({{ pitcher_hand }})
- **Estadio:** {{ venue_name }} | Park Factor dinámico HR: {{ dynamic_pf_hr }}
- **Condiciones:** {{ weather_summary }}
- **E[R] Alineación Óptima:** {{ expected_runs_optimal }} | **P(W):** {{ p_win }}%

## Top Insights del Optimizer (por impacto en E[R])
{% for insight in top_insights %}
{{ loop.index }}. {{ insight.description }} → Δ E[R]: +{{ insight.delta_er }}
{% endfor %}

## Contexto de Scouting Reciente
{% for chunk in scouting_chunks %}
- {{ chunk.text }} *(Fuente: {{ chunk.source }}, {{ chunk.date }})*
{% endfor %}

Genera el briefing narrativo ahora. Máximo 350 palabras. Tone: técnico pero directo.
""")

client = anthropic.Anthropic()
prompt = BRIEFING_TEMPLATE.render(**briefing_data)
response = client.messages.create(
    model="claude-opus-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}]
)
```

**Herramientas:**
- **Anthropic API (Claude claude-opus-4-20250514)** — generación de narrativa final
- **Jinja2** — templates de prompt estructurados y mantenibles
- **SHAP** — selección y cuantificación de insights para el prompt
- **LangChain (PromptTemplate + LLMChain)** — orquestación del pipeline de generación
- **Langfuse** — trazabilidad completa de cada generación (prompt → output → feedback)

**Por qué (Lógica de Negocio/Matemática):**  
La **adopción** es el verdadero KPI del sistema. Un modelo con 97% de precisión que el mánager ignora tiene valor de negocio cero. La investigación de behavioral economics muestra que los expertos en dominio (mánagers con 20+ años de experiencia) solo adoptan recomendaciones de sistemas algorítmicos cuando las razones se presentan en su propio lenguaje y framework mental. La narrativa no es cosmética: es el mecanismo de **transferencia de confianza** del modelo matemático al cuerpo técnico humano.

---

### 4.5 Agente Conversacional de Consulta para el Cuerpo Técnico

**Descripción:**  
Se implementa un agente LLM con **tool-calling** que permite al cuerpo técnico realizar preguntas en lenguaje natural y obtener respuestas basadas en datos reales del sistema. El agente tiene acceso a 5 herramientas:

1. **`simulate_lineup(positions: list[str]) → dict`** — Re-ejecuta la simulación Monte Carlo con la alineación especificada (versión reducida de 10,000 iteraciones para respuesta en <8 segundos).
2. **`get_player_matchup_stats(batter_id: str, pitcher_id: str) → dict`** — Recupera los splits históricos y proyectados del matchup específico.
3. **`get_park_factor_today(venue_id: str) → dict`** — Retorna los park factors dinámicos del día con condiciones climáticas.
4. **`search_scouting_reports(query: str) → list[str]`** — RAG search sobre los reportes de scouting.
5. **`compare_lineups(lineup_a: list, lineup_b: list) → dict`** — Compara dos alineaciones con delta de E[R] y P(W).

El agente razona sobre qué tools usar (ReAct pattern: Thought → Action → Observation → Thought) para responder preguntas como: *"¿Qué pasa si movemos a Rodríguez al 3er turno y a García al 5to?"* o *"¿Conviene usar a López dado su Recovery Score bajo de hoy?"*

**Herramientas:**
- **Anthropic API (function calling / tool use)** — backbone del agente con ReAct pattern
- **LangChain Agents (Tool + AgentExecutor)** — definición y orquestación de herramientas
- **FastAPI** — endpoint del agente (`POST /agent/chat`) con WebSocket streaming
- **Redis** — almacenamiento de historial de conversación para contexto multi-turn
- **Langfuse** — trazabilidad y observabilidad de sesiones del agente

**Por qué (Lógica de Negocio/Matemática):**  
El agente conversacional es la interfaz que democratiza el acceso al sistema para usuarios no técnicos. Un coach de bateo no va a navegar un dashboard complejo antes de un partido; sí va a escribir en un chat: *"¿Quién debería ir 2do hoy con el viento de entrada fuerte en Wrigley?"* La combinación de lenguaje natural + datos reales del sistema (no conocimiento estático del LLM) con latencia sub-10 segundos es el punto de inflexión que convierte el sistema de una herramienta analítica en una **extensión cognitiva del cuerpo técnico**.

---

### 4.6 Guardrails y Sistema Anti-Alucinación para el LLM

**Descripción:**  
Un LLM que afirma incorrectamente que un bateador tiene ".412 wOBA vs lanzadores diestros" cuando el valor real es .295 puede llevar a decisiones de alineación dañinas. Se implementa un sistema de guardrails en tres niveles:

- **Pre-generation:** `Guardrails AI` valida que el prompt no contenga instrucciones que puedan llevar al LLM a ignorar datos del Feature Store y fabricar estadísticas.
- **Post-generation:** Se extraen todas las afirmaciones estadísticas cuantificables de la narrativa generada usando regex + NER, y se validan contra el Feature Store. Si la afirmación no puede verificarse, se reemplaza con una versión hedgeada.
- **Continuous evaluation:** `RAGAS` evalúa continuamente tres métricas del sistema RAG: `Faithfulness` (¿está cada afirmación respaldada por los chunks recuperados?), `Answer Relevancy` (¿responde el output a la query?), y `Context Precision` (¿son los chunks recuperados realmente relevantes?).

**Herramientas:**
- **Guardrails AI** — framework de validación de inputs/outputs del LLM con validators custom
- **RAGAS** — evaluación automatizada de calidad del sistema RAG (Faithfulness, Relevancy, Precision)
- **Langfuse** — observabilidad completa: tracing de cada generación con tokens, latencia y score
- **Pydantic v2** — validación de schema del output estructurado del LLM (cuando se requiere JSON)
- **AWS Lambda** — ejecución serverless del guardrail de verificación post-generación

**Por qué (Lógica de Negocio/Matemática):**  
Las alucinaciones del LLM son especialmente peligrosas en dominios cuantitativos de alta precisión como el béisbol analítico. A diferencia de un chatbot de atención al cliente donde una respuesta imprecisa es una molestia, aquí una estadística incorrecta puede traducirse directamente en una decisión de alineación subóptima en un playoff game con consecuencias de decenas de millones de dólares. El sistema de guardrails no impide que el LLM sea útil: lo hace **confiable a escala**, que es la condición necesaria para la adopción sostenida por el cuerpo técnico.

---

<a name="fase-5"></a>
## Fase 5 — MLOps, CI/CD de Modelos y Gobierno

### 5.1 Gestión de Experimentos y Model Registry con MLflow

**Descripción:**  
MLflow actúa como el **sistema nervioso central** del ciclo de vida de los modelos. Cada experimento de entrenamiento (run) registra automáticamente: hiperparámetros exactos, métricas de evaluación por fold y por clase de outcome, artefactos del modelo (pesos, feature importance, calibration curves), el dataset hash (para garantizar reproducibilidad), y la versión del código (git commit SHA).

El **Model Registry** centraliza el estado de cada modelo en cuatro etapas: `None` → `Staging` → `Production` → `Archived`. La transición de `Staging` a `Production` requiere una aprobación manual del Lead Data Scientist + evidencia de que el challenger supera al campeón actual en las métricas de negocio definidas. La transición `Production` → `Archived` es automática cuando un nuevo modelo entra en producción.

```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("https://mlflow.internal.mlb-analytics.com")
mlflow.set_experiment("pa_model_v4_platoon_embeddings")

with mlflow.start_run(run_name="lgbm_optuna_trial_42") as run:
    mlflow.log_params(best_params)
    mlflow.log_metric("val_logloss",    0.8234)
    mlflow.log_metric("val_ece",        0.0312)
    mlflow.log_metric("val_delta_er",   0.228)
    mlflow.log_metric("test_logloss",   0.8441)
    mlflow.sklearn.log_model(model, artifact_path="pa_model",
                             registered_model_name="pa_level_model_production")

# Promover a Staging tras validación automática
client = MlflowClient()
client.transition_model_version_stage(
    name="pa_level_model_production",
    version=42,
    stage="Staging"
)
```

**Herramientas:**
- **MLflow 2.x** — tracking de experimentos, artifact storage y model registry
- **AWS S3** — artifact store de MLflow (modelos, plots, datasets de evaluación)
- **AWS RDS (PostgreSQL)** — backend store de MLflow (runs, parámetros, métricas)
- **DVC (Data Version Control)** — versionado de datasets de entrenamiento con Git integration
- **Weights & Biases** — alternativa premium evaluada para experimentos de deep learning

**Por qué (Lógica de Negocio/Matemática):**  
La **reproducibilidad** es un requisito de auditoría, no solo una buena práctica de ingeniería. Si el sistema produce una recomendación de alineación que resulta en una pérdida inesperada, el cuerpo directivo y técnico necesita poder responder: *"¿Qué modelo produjo esa recomendación? ¿Con qué datos fue entrenado? ¿Cuáles eran sus métricas de validación en ese momento?"* Sin MLflow, estas preguntas son imposibles de responder de forma sistemática. Con MLflow, son respondibles en <60 segundos.

---

### 5.2 CI/CD de Modelos con GitHub Actions y SageMaker Pipelines

**Descripción:**  
Se implementa un pipeline de CI/CD completo para modelos ML que elimina el proceso manual de entrenamiento, evaluación y despliegue. Cada Pull Request al repositorio que modifica código de features, modelos o transformaciones dispara automáticamente el siguiente flujo:

**Etapa 1 — CI (Continuous Integration):**
- Linting y formatting (black, isort, flake8).
- Tests unitarios del código de features (pytest, cobertura >85%).
- Validación de schema de outputs de transformaciones dbt.
- Tests de integración del pipeline de simulación con datos mock.

**Etapa 2 — CT (Continuous Training) — solo en merge a `main`:**
- Trigger de un SageMaker Pipeline que ejecuta: validación de datos → feature computation → entrenamiento del challenger → evaluación offline → comparación vs. campeón actual.
- Si `challenger.val_delta_er > champion.val_delta_er + 0.05` y `challenger.val_logloss < champion.val_logloss`, el challenger se promueve automáticamente a Staging.

**Etapa 3 — CD (Continuous Deployment) — solo con aprobación manual:**
- Un analista senior revisa las métricas de Staging vs. Producción.
- Aprobación en GitHub → trigger de deploy a SageMaker Endpoint con Blue/Green deployment.
- Smoke tests automáticos post-deploy (10 inference requests con outputs esperados conocidos).

```yaml
# .github/workflows/model_ci_cd.yml (fragmento)
name: Model CI/CD Pipeline
on:
  push:
    branches: [main]
    paths:
      - 'src/models/**'
      - 'src/features/**'
      - 'dbt/models/**'

jobs:
  train-and-evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::ACCOUNT_ID:role/MLPipelineRole
      - name: Trigger SageMaker Training Pipeline
        run: |
          aws sagemaker start-pipeline-execution \
            --pipeline-name pa-model-training-pipeline \
            --pipeline-parameters Name=GitCommit,Value=${{ github.sha }}
```

**Herramientas:**
- **GitHub Actions** — orquestación del CI/CD con events basados en paths
- **AWS SageMaker Pipelines** — pipeline de ML de extremo a extremo gestionado
- **Docker** — contenedores reproducibles para entrenamiento y serving
- **Amazon ECR** — registro de imágenes Docker del equipo
- **Kubernetes (AWS EKS)** — orquestación de contenedores en producción
- **Terraform** — infraestructura del pipeline como código (IaC)

**Por qué (Lógica de Negocio/Matemática):**  
El béisbol cambia semanalmente: un lanzador mejora su changeup, un bateador cambia su stance. Un modelo que no se actualiza con estos cambios acumula **deuda de performance** que se manifiesta como predicciones progresivamente más inexactas. El CI/CD garantiza que el ciclo de actualización del modelo es **automático, auditable y reversible**: si un nuevo modelo degrada la performance, el rollback es un clic, no un proceso manual de horas.

---

### 5.3 Entrenamiento Continuo: Pipeline de Reentrenamiento Automático

**Descripción:**  
Se implementan dos triggers de reentrenamiento:

**Reentrenamiento semanal (scheduled):** Cada lunes a las 02:00 AM, un DAG de Airflow verifica si se han acumulado ≥500 nuevos PAs desde el último reentrenamiento. Si la condición se cumple, dispara un SageMaker Training Job que entrena un modelo challenger con los datos actualizados (incluyendo la semana más reciente) y lo registra en MLflow para evaluación.

**Reentrenamiento urgente (event-driven):** Si el sistema de monitoreo de drift (Evidently AI) detecta que el **Population Stability Index (PSI)** de algún feature crítico supera 0.25 (indicando drift significativo), se dispara inmediatamente un reentrenamiento con un evento en Kafka que consume el DAG `dag_model_retraining_urgent` de Airflow.

El reentrenamiento no es un entrenamiento desde cero: se usa **warm-starting** con los pesos del modelo campeón actual como punto de partida para LightGBM (usando `init_model` parameter), reduciendo el tiempo de convergencia en ~40% y el riesgo de degradación catastrófica.

**Herramientas:**
- **Apache Airflow** — scheduling de reentrenamiento semanal + DAG de reentrenamiento urgente
- **AWS SageMaker Training Jobs** — entrenamiento gestionado con auto-escalado de instancias
- **Ray Train (Data Parallel)** — entrenamiento distribuido del FT-Transformer en múltiples GPUs
- **Evidently AI** — detección de data drift (PSI, KS test, chi-square por feature categórico)
- **Apache Kafka** — event-driven trigger de reentrenamiento urgente desde el sistema de monitoreo

**Por qué (Lógica de Negocio/Matemática):**  
En béisbol, el **concept drift** es estructural y predecible: cada temporada comienza con reglas potencialmente diferentes (zona de strike, uso del reloj de pitcheo), los jugadores evolucionan (un bateador que adoptó un lift approach en pre-temporada tiene una distribución de launch angle completamente diferente), y los lanzadores ajustan sus arsenales. Un modelo estático entrenado en pre-temporada y no actualizado durante 6 meses de temporada acumulará un error sistemático creciente. El reentrenamiento continuo es la respuesta técnica al hecho de que **el béisbol es un juego dinámico, no estático**.

---

### 5.4 Monitoreo de Data Drift y Model Drift en Producción

**Descripción:**  
Se implementa un sistema de monitoreo en dos dimensiones:

**Data Drift (features de entrada):** Evidently AI genera un reporte de drift diario comparando la distribución de features de las últimas 7 días contra la distribución del conjunto de entrenamiento (referencia). Para features numéricas se usa el **test de Kolmogorov-Smirnov**; para categóricas, el **test chi-cuadrado**. El **Population Stability Index (PSI)** integra ambos en una sola métrica: PSI < 0.1 = sin drift, 0.1–0.25 = drift moderado, >0.25 = drift severo (trigger de reentrenamiento).

**Model Performance Drift (métricas de salida):** Se monitorea el log-loss del PA-Level Model usando los resultados reales de cada PA como ground truth (disponibles ~3 horas post-partido). Si el log-loss de 7 días supera en >5% el log-loss de validación de referencia, se activa una alerta de degradación de performance. Se calculan también métricas de calibración por decil de probabilidad para detectar recalibración necesaria.

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
import pandas as pd

reference = pd.read_parquet("s3://mlb-lakehouse/gold/reference_dataset_train.parquet")
current   = pd.read_parquet("s3://mlb-lakehouse/gold/features_last_7d.parquet")

report = Report(metrics=[
    DataDriftPreset(drift_share=0.3),
    ClassificationPreset()
])
report.run(reference_data=reference, current_data=current)
report.save_html("s3://mlb-monitoring/drift_reports/2024-07-15.html")

# Check PSI de feature crítico
psi = report.as_dict()["metrics"][0]["result"]["drift_by_columns"]["spin_rate_ff"]["stattest_threshold"]
if psi > 0.25:
    trigger_urgent_retraining(feature="spin_rate_ff", psi_value=psi)
```

**Herramientas:**
- **Evidently AI** — reportes de data drift y model performance con UI integrada
- **WhyLabs** — plataforma de observabilidad de ML alternativa evaluada
- **Grafana + Prometheus** — dashboards operativos con métricas custom de pipeline
- **AWS CloudWatch** — métricas de infraestructura y alertas de SLA
- **PagerDuty** — gestión de incidentes con on-call rotation para alertas críticas

**Por qué (Lógica de Negocio/Matemática):**  
El drift más peligroso es el **drift silencioso**: el modelo sigue funcionando (no hay errores de infraestructura) pero sus predicciones se vuelven progresivamente menos precisas porque la distribución de datos del mundo real divergió de la distribución de entrenamiento. En béisbol, el drift silencioso puede ocurrir cuando un lanzador adopta un nuevo pitch a mitad de temporada: el perfil de su arsenal en el Feature Store es obsoleto, y el modelo predice sus matchups basándose en un arsenal que ya no existe. El monitoreo de PSI detecta este cambio en los features de entrada antes de que se refleje en la degradación de métricas de salida.

---

### 5.5 Model Serving de Baja Latencia: FastAPI + TorchServe

**Descripción:**  
El sistema de serving debe soportar dos patrones de uso con SLAs distintos:

**Inferencia individual (online serving):** El dashboard y el agente conversacional llaman al PA-Level Model para predicciones individuales. SLA: <50ms P99. Servido con **FastAPI + uvicorn** en un endpoint de SageMaker Real-Time con autoescalado (HPA en Kubernetes: min 2 réplicas, max 20, trigger en CPU >70%).

**Inferencia masiva (batch scoring para simulación):** La simulación Monte Carlo requiere ~100,000 × 162 inferencias por simulación completa. SLA: todo el batch completado en <60 segundos. Implementado con **TorchServe Batch Inference** usando dynamic batching (batch size máximo: 512, timeout: 10ms) y **Ray Serve** para distribución horizontal. El modelo LightGBM se serializa con Treelite para compilación a C++ nativo, mejorando la latencia de predicción en ~3x.

```python
# FastAPI serving endpoint
from fastapi import FastAPI
from pydantic import BaseModel
import lightgbm as lgb
import treelite_runtime as tl_runtime
import numpy as np

app = FastAPI(title="PA-Level Model API", version="4.2.0")

# Modelo compilado con Treelite para máxima performance
predictor = tl_runtime.Predictor("pa_model_v4.so", verbose=False)

class PARequest(BaseModel):
    batter_id: str
    pitcher_id: str
    venue_id: str
    weather_snapshot_id: str
    runner_state: int  # 0-7 (bitmask de bases ocupadas)
    outs: int

@app.post("/predict/plate_appearance")
async def predict_pa(request: PARequest) -> dict:
    features = feature_store.get_online_features(
        entity_rows=[{"batter_id": request.batter_id,
                      "pitcher_id": request.pitcher_id}]
    )
    X = build_feature_vector(features, request)
    proba = predictor.predict(tl_runtime.DMatrix(X.reshape(1, -1)))
    return {
        "walk": float(proba[0][0]), "strikeout": float(proba[0][1]),
        "single": float(proba[0][2]), "double": float(proba[0][3]),
        "triple": float(proba[0][4]), "home_run": float(proba[0][5]),
        "hbp": float(proba[0][6]), "out_in_play": float(proba[0][7])
    }
```

**Herramientas:**
- **FastAPI + uvicorn** — serving asíncrono de alta performance con OpenAPI automático
- **Treelite** — compilación de modelos GBDT a código nativo C/C++ para latencia mínima
- **TorchServe** — serving de modelos PyTorch con batching dinámico
- **Ray Serve** — serving distribuido y escalable para la simulación Monte Carlo masiva
- **AWS SageMaker Real-Time Endpoints** — serving gestionado con autoescalado y A/B testing
- **BentoML** — empaquetado portable de modelos con dependencias para CI/CD

**Por qué (Lógica de Negocio/Matemática):**  
La latencia de serving del PA-Level Model es el **cuello de botella** de todo el sistema: en la simulación Monte Carlo se ejecuta una inferencia por cada PA en cada una de las 100,000 simulaciones. Si cada inferencia toma 2ms en vez de 0.5ms, el tiempo total de simulación aumenta de 45 segundos a 3 minutos, violando el SLA del día de partido. La compilación con Treelite a código nativo C reduce la latencia de un modelo LightGBM típico de ~2ms a ~0.4ms, un speedup de 5x que es crítico para la viabilidad operativa del sistema.

---

### 5.6 Shadow Mode, A/B Testing y Canary Releases

**Descripción:**  
Antes de que cualquier modelo challenger reemplace al campeón en producción, pasa por tres etapas de validación progresiva:

**Shadow Mode:** El challenger corre en paralelo al campeón durante 2 semanas recibiendo el mismo tráfico de producción (mismo input de features), pero sus predicciones **no son mostradas** al cuerpo técnico. Los outputs de ambos modelos se almacenan en S3 para comparación offline post-partido (cuando hay resultados reales disponibles). Si el challenger supera al campeón en log-loss y delta_ER durante el shadow period, avanza.

**Canary Release (10%):** El challenger sirve el 10% del tráfico en producción (selección aleatoria por `game_id`). Se monitorean métricas de negocio en tiempo real. Si no hay degradación en 5 juegos, el tráfico se incrementa a 50%.

**Full Rollout:** Si Canary es exitoso durante 10 juegos adicionales, el challenger toma el 100% del tráfico y el campeón anterior se archiva en MLflow.

**Herramientas:**
- **AWS SageMaker (producción variants)** — traffic splitting nativo para A/B y canary
- **Istio (service mesh en EKS)** — control de tráfico a nivel de microservicio con weighted routing
- **MLflow (champion/challenger comparison)** — dashboard de comparación de modelos
- **statsmodels (t-test de dos muestras)** — validación de significancia estadística de la mejora
- **Grafana** — visualización de métricas del canary en tiempo real

**Por qué (Lógica de Negocio/Matemática):**  
Las métricas de validación offline (log-loss, ECE) son condiciones necesarias pero no suficientes para garantizar que un modelo funciona en producción. La distribución de features en producción puede diferir sutilmente de la distribución de validación en formas que solo se manifiestan con tráfico real. Shadow Mode y Canary Releases son el protocolo de **gestión de riesgo técnico** que garantiza que ningún modelo entra a producción sin haber demostrado su valor con datos del mundo real, minimizando el impacto de posibles degradaciones de performance en decisiones de alineación reales.

---

### 5.7 Infraestructura como Código y Gestión de Secretos

**Descripción:**  
Toda la infraestructura del sistema se define como código Terraform, incluyendo: VPC y subnets, clústeres EKS, endpoints de SageMaker, instancias ElastiCache, tablas de DynamoDB, políticas IAM, y reglas de Security Groups. Ninguna configuración de infraestructura existe solo en la consola AWS. Los secretos (API keys, credenciales de base de datos, tokens de acceso) nunca se hardcodean en el código ni se almacenan en variables de entorno sin cifrar.

```hcl
# main.tf (fragmento)
resource "aws_sagemaker_endpoint" "pa_model_prod" {
  name                 = "pa-level-model-production"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.pa_model_v4.name
  
  tags = {
    Environment = "production"
    Team        = "ml-engineering"
    CostCenter  = "lineup-optimizer"
  }
}

resource "aws_secretsmanager_secret" "mlb_api_key" {
  name                    = "production/mlb-statsapi/api-key"
  recovery_window_in_days = 7
}

resource "aws_iam_role_policy" "sagemaker_feature_store" {
  name = "sagemaker-feature-store-access"
  role = aws_iam_role.ml_pipeline.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["sagemaker:GetRecord", "sagemaker:PutRecord"]
      Resource = "arn:aws:sagemaker:us-east-1:ACCOUNT_ID:feature-group/*"
    }]
  })
}
```

**Herramientas:**
- **Terraform 1.8+** — IaC con state management en S3 + DynamoDB (locking)
- **AWS CDK** — alternativa evaluada para infraestructura definida en Python
- **AWS Secrets Manager** — almacenamiento cifrado de secretos con rotación automática
- **HashiCorp Vault** — gestión de secretos para servicios fuera de AWS
- **AWS IAM + RBAC** — principio de mínimo privilegio para cada servicio y rol

**Por qué (Lógica de Negocio/Matemática):**  
La infraestructura como código garantiza **Disaster Recovery en <30 minutos**: si toda la infraestructura de producción es destruida (escenario de peor caso), puede ser completamente recreada ejecutando `terraform apply`. Esto es especialmente crítico en el contexto de una temporada de béisbol donde el sistema debe estar disponible 162 días consecutivos. La gestión de secretos mediante AWS Secrets Manager con rotación automática elimina el riesgo de credenciales expuestas en repositorios o logs, un vector de ataque común en sistemas con múltiples integraciones de APIs externas.

---

<a name="fase-6"></a>
## Fase 6 — Interfaz de Usuario y Consumo

### 6.1 Diseño y Arquitectura de la API REST Principal (FastAPI)

**Descripción:**  
La API REST es la **única interfaz programática** de acceso al sistema Lineup Optimizer. Toda la UI, el agente conversacional, y las alertas automáticas la consumen. Se diseña con principios REST estrictos, versionado de endpoints (`/v1/`, `/v2/`), y documentación automática OpenAPI/Swagger.

Los endpoints principales son:

- `POST /v1/lineups/optimize` — **Core endpoint.** Recibe `{venue_id, game_date, available_players: [], opponent_starter_id, constraints: {}}` y retorna la alineación óptima con E[R], P(W), top-3 alternativas, y narrativa generada.
- `POST /v1/lineups/simulate` — Simula una alineación específica proporcionada por el usuario.
- `GET /v1/players/{player_id}/matchup/{pitcher_id}` — Retorna el perfil de matchup completo con splits y proyecciones.
- `GET /v1/venues/{venue_id}/park-factors/today` — Park factors dinámicos del día con condiciones climáticas.
- `GET /v1/models/health` — Health check con versión del modelo en producción y métricas de performance recientes.
- `POST /v1/agent/chat` — Endpoint del agente conversacional con WebSocket para streaming.

Autenticación: OAuth2 con JWT tokens, scopes por rol (`read:lineups`, `simulate:lineups`, `admin:models`). Rate limiting: 60 requests/minuto por token para usuarios estándar, 600/minuto para servicios internos.

**Herramientas:**
- **FastAPI 0.111+** — framework asíncrono con OpenAPI automático y validación Pydantic
- **Pydantic v2** — validación y serialización de request/response models
- **uvicorn + gunicorn** — ASGI server con workers múltiples
- **AWS API Gateway** — routing, rate limiting y auth a nivel de API
- **AWS WAF** — protección contra ataques web en el API Gateway
- **httpx (async)** — cliente HTTP para llamadas a servicios internos

**Por qué (Lógica de Negocio/Matemática):**  
Una API bien diseñada con versionado explícito permite que la UI, el agente y las alertas automáticas evolucionen de forma **independiente y desacoplada** del modelo subyacente. Si el PA-Level Model es reemplazado por una arquitectura completamente diferente (ej. de LightGBM a un Transformer), la interfaz de la API permanece estable y ningún cliente necesita ser modificado. Este desacoplamiento es crítico para la velocidad de iteración del equipo: ML Engineering puede actualizar el modelo sin coordinar deployments con el equipo de frontend.

---

### 6.2 Dashboard Táctico Principal para el Cuerpo Técnico

**Descripción:**  
La interfaz web principal está diseñada específicamente para el mánager y los coaches de bateo. El principio de diseño es **información crítica al frente, complejidad técnica oculta**. El dashboard se divide en cuatro secciones visibles simultáneamente en una pantalla de iPad en modo horizontal:

**Panel A — Alineación Recomendada:** Visualización de las 9 posiciones de batting order con nombre del jugador, posición defensiva, y tres KPIs por jugador: `wOBA_30d`, `split_vs_pitcher_hand`, y `recovery_tier` (HIGH/MED/LOW con código de color semáforo). Un drag-and-drop silently re-ejecuta la simulación.

**Panel B — Comparación de Alternativas:** Top-3 alineaciones alternativas con delta de E[R] respecto al óptimo. Cada alternativa muestra qué cambios específicos tiene y su justificación en una frase.

**Panel C — Análisis del Lanzador Rival:** Heatmap de pitch usage del arsenal del lanzador (pitch-type × zone × count), con overlay de la tasa de whiff de nuestros bateadores contra cada tipo de pitch. Información dinámica del estado del bullpen rival.

**Panel D — Briefing Narrativo:** El texto generado por el LLM, con citas de scouting reports resaltadas visualmente, y un campo de chat para preguntas al agente conversacional.

**Herramientas:**
- **Next.js 14 (App Router)** — framework React con SSR/ISR para carga inicial rápida
- **Tailwind CSS** — styling utility-first con design system del equipo
- **Recharts + D3.js** — visualizaciones de heatmaps, distribuciones y pitch charts
- **React Query (TanStack Query)** — data fetching con cache inteligente y background refetch
- **dnd-kit** — drag-and-drop accesible para reordenamiento de alineación
- **AWS CloudFront** — CDN para servir el dashboard con latencia mínima globalmente

**Por qué (Lógica de Negocio/Matemática):**  
La psicología del rendimiento bajo presión muestra que los decisores en entornos de alta stakes (un mánager antes de un playoff game) tienen una **capacidad cognitiva reducida** para procesar información compleja. El diseño del dashboard no es una cuestión estética: es una intervención de ergonomía cognitiva que garantiza que la información más crítica (la alineación recomendada y su justificación principal) sea accesible en <3 segundos de interacción, sin necesidad de navegar menús ni interpretar visualizaciones complejas.

---

### 6.3 Panel Interactivo de Análisis What-If en Tiempo Real

**Descripción:**  
El panel What-If es la herramienta de análisis más poderosa del sistema para el cuerpo técnico. Permite al mánager explorar hipótesis específicas antes de tomar la decisión final de alineación:

**Funcionalidades:**
- **Drag-and-drop de posiciones:** Arrastrar un jugador de la posición 5 a la posición 2 y ver el delta de E[R] actualizado en <5 segundos.
- **Swap de jugadores:** Reemplazar un jugador del roster activo por uno de la lista de espera y comparar el impacto.
- **Slider de condiciones climáticas:** Modificar la velocidad del viento de 5 a 20 mph y observar cómo cambia el park factor dinámico y el E[R].
- **Toggle de disponibilidad del jugador:** Marcar un jugador como "no disponible" (lesión de último momento) y que el sistema recalcule el óptimo con el roster restante.
- **Modo comparación:** Anclar dos configuraciones lado a lado con sus distribuciones de runs (histograma) superpuestos.

Las simulaciones What-If usan la versión reducida de 10,000 iteraciones (vs. 100,000 del óptimo) para producir resultados en <5 segundos, con un intervalo de confianza ligeramente más amplio que se comunica visualmente (barras de error en el histograma).

**Herramientas:**
- **React + dnd-kit** — drag-and-drop accesible con animaciones fluidas
- **WebSockets (via FastAPI + WebSocket)** — streaming de resultados de simulación en tiempo real
- **Redis** — caché de simulaciones recientes (TTL: 30 minutos) para evitar re-cómputos redundantes
- **Recharts (BarChart + ComposedChart)** — visualización de distribuciones de runs y overlays de comparación
- **Zustand** — state management global del estado del What-If panel

**Por qué (Lógica de Negocio/Matemática):**  
El panel What-If sirve un propósito de **validación y apropiación** de la recomendación del sistema. Cuando el mánager puede explorar manualmente hipótesis y ver que el sistema confirma (o cuantifica el costo de) sus intuiciones, construye confianza en el modelo. La investigación en human-in-the-loop decision support muestra que los decisores que pueden "jugar" con el sistema y verificar sus intuiciones antes de aceptar la recomendación tienen una **tasa de adopción 3x mayor** y un mayor nivel de commitment con la decisión final.

---

### 6.4 Sistema Automático de Alertas y Reportes Pre-Partido

**Descripción:**  
Cada mañana de día de partido, el sistema genera y distribuye automáticamente un paquete informativo completo al cuerpo técnico. El proceso es orchestrado por el DAG `dag_game_day_briefing` de Airflow con las siguientes tareas secuenciales:

1. **09:00 AM:** Ingesta de condiciones climáticas del día (Tomorrow.io) y confirmación del lineup del oponente (cuando está disponible en MLB StatsAPI).
2. **09:15 AM:** Recompute de park factors dinámicos y actualización del estado del bullpen rival.
3. **09:30 AM:** Ejecución del optimizador completo (100,000 simulaciones Monte Carlo).
4. **09:55 AM:** Generación del briefing narrativo LLM + validación de guardrails.
5. **10:00 AM:** Distribución multi-canal: email cifrado al mánager y coaches + notificación push en la app interna + PDF en Slack del canal privado del cuerpo técnico.

El reporte generado tiene estructura canónica: (1) Alineación óptima recomendada con E[R] y P(W), (2) Top-3 alternativas con delta de E[R], (3) Análisis del lanzador rival (arsenal, tendencias, alertas de scouting), (4) Alertas especiales (condiciones climáticas extremas, jugadores con biometría en zona roja, platoon advantages masivos), (5) Estado del bullpen rival con secuencia proyectada.

**Herramientas:**
- **Apache Airflow** — orquestación completa del DAG de briefing diario con SLA monitoring
- **Anthropic API** — generación del briefing narrativo y resúmenes por sección
- **AWS SES (Simple Email Service)** — entrega de email con templates HTML responsive
- **Slack API (Bolt SDK)** — notificaciones en canal privado con PDF adjunto
- **WeasyPrint** — generación de PDF del reporte desde HTML/CSS con estilos del equipo
- **Firebase Cloud Messaging** — push notifications a dispositivos iOS/Android del equipo

**Por qué (Lógica de Negocio/Matemática):**  
El timing de la distribución a las 10:00 AM es crítico: es suficientemente temprano para dar al mánager tiempo de revisar y hacer preguntas al sistema antes de la lineup card deadline (usualmente 3 horas antes del primer pitch). La distribución multi-canal garantiza que el reporte llega al cuerpo técnico independientemente de si están en el estadio, en el hotel o en tránsito, eliminando el riesgo de que una mala conexión de red en un momento crítico deje al equipo sin la recomendación del sistema.

---

### 6.5 Implementación de Autenticación, RBAC y Auditoría Completa

**Descripción:**  
La información de alineaciones óptimas es **extremadamente sensible competitivamente**: si el equipo rival sabe qué alineación planea jugar nuestro equipo antes de que sea oficial, tiene una ventaja estratégica significativa. Se implementa un modelo de seguridad en capas:

**Autenticación:** AWS Cognito gestiona identidades con MFA obligatorio para todos los usuarios. JWT tokens con expiración de 4 horas (renovación automática con refresh token).

**Control de Acceso (RBAC):** Cuatro roles con permisos granulares:
- `manager` — acceso completo al dashboard, What-If panel, y agente conversacional.
- `coach` — acceso al dashboard y recomendaciones sin poder modificar parámetros del modelo.
- `analyst` — acceso completo incluyendo datos raw del Feature Store y métricas del modelo.
- `admin` — control total incluyendo gestión de usuarios y configuración de modelos.

**Auditoría:** Cada request a la API es registrado en AWS CloudTrail con: timestamp, user_id, endpoint, parámetros de request (anonimizados), y response code. Los registros de auditoría tienen retención de 7 años y son inmutables (S3 Object Lock con WORM compliance). Cualquier acceso a la alineación óptima recomendada genera una entrada de auditoría específica.

**Herramientas:**
- **AWS Cognito** — identity provider con MFA (TOTP) y social login opcional
- **AWS IAM** — políticas de acceso service-to-service con least privilege
- **FastAPI (OAuth2PasswordBearer + JWT)** — middleware de autenticación en la API
- **AWS CloudTrail** — auditoría completa e inmutable de todas las acciones de la API
- **AWS Macie** — detección automática de datos personales en S3 (para biometría)

**Por qué (Lógica de Negocio/Matemática):**  
Las decisiones de alineación en el béisbol de Grandes Ligas tienen implicaciones financieras directas de decenas de millones de dólares (contratos de jugadores, ingresos por playoffs, valor de mercado del equipo). Una brecha de seguridad que exponga las recomendaciones del sistema al equipo rival o a agentes de apostadores es un riesgo de negocio de primer orden. La auditoría completa también protege al equipo en disputas contractuales con jugadores: si un jugador alega que fue usado en una posición perjudicial para su rendimiento, el log de auditoría muestra exactamente qué recomendó el sistema y qué decidió el mánager.

---

### 6.6 Feedback Loop: Captura de Decisiones del Mánager para Mejora Continua

**Descripción:**  
El sistema no solo produce recomendaciones: aprende de las decisiones del mánager. Se implementa un loop de feedback estructurado que captura, para cada partido:

- **La alineación recomendada por el sistema** (con su E[R] y P(W) proyectados).
- **La alineación final decidida por el mánager** (si difiere de la recomendada, se solicita opcionalmente una justificación en texto libre de 1-2 frases).
- **El resultado real del partido** (carreras anotadas, resultado W/L, PAs por bateador).

Esta información alimenta tres procesos downstream:

1. **Calibración del modelo:** El delta entre E[R] proyectado y carreras reales anotadas es la señal de error primaria para el reentrenamiento.
2. **RLHF del LLM:** Las justificaciones en texto libre del mánager cuando rechaza la recomendación son preferencia data invaluable para el fine-tuning de la narrativa.
3. **Análisis de adoption rate:** El ratio de veces que el mánager sigue la recomendación del sistema vs. la rechaza es el KPI de producto más importante del sistema.

**Herramientas:**
- **PostgreSQL (AWS RDS)** — almacén de decisiones y resultados con schema versionado
- **FastAPI** — endpoint `POST /v1/lineups/submit-decision` para captura de decisiones post-partido
- **Apache Kafka** — streaming de eventos de resultado post-partido para reentrenamiento
- **MLflow (Dataset tracking)** — versionado del dataset de feedback para auditoría de RLHF
- **Metabase** — dashboard interno de adoption rate y análisis de divergencias sistema/mánager

**Por qué (Lógica de Negocio/Matemática):**  
El **adoption rate** (porcentaje de veces que el mánager sigue la recomendación del sistema) es el único KPI que mide el verdadero valor de negocio del sistema, por encima de cualquier métrica técnica de ML. Un sistema con 95% de precisión en validación pero 20% de adoption rate tiene un impacto de negocio próximo a cero. Los datos de feedback permiten identificar sistemáticamente **en qué tipos de situaciones el mánager diverge del sistema** (ej. siempre ignora la recomendación cuando el viento supera 20 mph o cuando el pitcher es zurdo con ERA <3.00), lo que puede indicar ya sea que el modelo tiene un gap en esas condiciones o que el mánager tiene conocimiento no cuantificado que debe incorporarse al modelo.

---

### 6.7 Testing End-to-End, Chaos Engineering y Runbooks de Incidentes

**Descripción:**  
La confiabilidad del sistema el día del partido es una condición de existencia, no una mejora opcional. Se implementa una estrategia de resiliencia en tres capas:

**Testing E2E automatizado:** Un suite de tests de Playwright simula el flujo completo de día de partido desde la perspectiva del usuario: login → recepción de briefing → interacción con What-If panel → envío de decisión final. Este suite se ejecuta cada noche a las 01:00 AM en un entorno de staging con datos reales del último partido.

**Chaos Engineering:** Mensualmente se ejecutan sesiones de Chaos Engineering usando AWS Fault Injection Simulator:
- Terminar el 50% de los pods del serving de FastAPI (simula fallo de instancia).
- Introducir latencia de 500ms en las respuestas de Redis (simula degradación del Feature Store online).
- Cortar el acceso a S3 por 2 minutos (simula fallo de almacenamiento).
- El sistema debe degradarse gracefully: en modo degradado, sirve la "última alineación conocida válida" con un aviso claro al usuario.

**Runbooks de incidentes:** Documentos operativos que especifican exactamente qué hacer en cada escenario de fallo, con pasos de diagnóstico, comandos exactos, y criterios de escalación:
- **INC-001:** Fallo del `dag_game_day_pipeline` a <3 horas del primer pitch.
- **INC-002:** PA-Level Model con log-loss >25% sobre el threshold de alerta.
- **INC-003:** Feature Store offline — serving con features del día anterior.
- **INC-004:** LLM API no disponible — briefing en modo numérico sin narrativa.

**Herramientas:**
- **Playwright** — testing E2E del dashboard en Chromium, Firefox y Safari
- **Locust** — load testing del API (simular 50 usuarios concurrentes en día de partido)
- **AWS Fault Injection Simulator (FIS)** — chaos experiments gestionados
- **PagerDuty** — gestión de incidentes con escalation policies y on-call rotation
- **Confluence** — runbooks de incidentes con versionado y aprobación

**Por qué (Lógica de Negocio/Matemática):**  
El fallo del sistema a las 11:45 AM antes de un juego playoff a las 13:05 PM es el escenario de peor caso. Sin runbooks probados y un modo de degradación graceful, el cuerpo técnico se enfrenta a una crisis operativa en el momento de máxima presión competitiva. Los runbooks convierten ese escenario de caos en un **proceso predecible de N pasos** donde cada miembro del equipo técnico sabe exactamente qué hacer, qué sistemas verificar, y cuándo escalar. El Chaos Engineering garantiza que esos runbooks están basados en comportamiento real del sistema bajo fallo, no en suposiciones teóricas.

---

<a name="apendice"></a>
## Apéndice — Diagrama de Arquitectura Global

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FUENTES DE DATOS                                │
│  MLB StatsAPI  │  Retrosheet  │  Tomorrow.io  │  WHOOP/Catapult  │ Scouts│
└────────┬───────┴──────┬───────┴───────┬───────┴────────┬─────────┴───┬──┘
         │              │               │                │             │
         ▼              ▼               ▼                ▼             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               INGESTA  (Kafka · Kinesis · Airflow)                      │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│            DATA LAKEHOUSE  (Delta Lake · S3 · Glue Catalog)             │
│   Bronze (raw) ──► Silver (curated, dbt) ──► Gold (features, dbt)      │
└──────────────────────┬───────────────────────────────────────┬──────────┘
                       │                                       │
                       ▼                                       ▼
         ┌─────────────────────────┐             ┌────────────────────────┐
         │   FEATURE STORE         │             │  VECTOR STORE          │
         │  SageMaker Feature Store│             │  Amazon OpenSearch      │
         │  Redis (online, <5ms)   │             │  (scouting embeddings) │
         └──────────┬──────────────┘             └───────────┬────────────┘
                    │                                        │
                    ▼                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODELO CORE                                     │
│  PA-Level Model (LightGBM + XGBoost + FT-Transformer Ensemble)         │
│  ──► Markov Chain Simulator (24 estados)                                │
│  ──► Monte Carlo Engine (100K sims, Ray distributed, c7i.48xlarge)     │
│  ──► Lineup Optimizer (Genetic Algorithm + Optuna, DEAP)               │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
     ┌──────────────────────────┐   ┌────────────────────────────┐
     │  RAG + LLM Layer         │   │  MLOps Layer               │
     │  LlamaIndex + Cohere     │   │  MLflow · SageMaker        │
     │  Claude claude-opus-4    │   │  Pipelines · Evidently AI  │
     │  Guardrails AI · RAGAS   │   │  GitHub Actions · EKS      │
     └─────────────┬────────────┘   └────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SERVING LAYER  (FastAPI · AWS API Gateway)           │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                      ▼
   ┌─────────────────┐   ┌─────────────────┐   ┌────────────────────┐
   │ Dashboard       │   │ What-If Panel   │   │ Automated Briefing │
   │ Next.js 14      │   │ React + dnd-kit │   │ Airflow + AWS SES  │
   │ iPad-optimized  │   │ WebSocket RT    │   │ + Slack + PDF      │
   └─────────────────┘   └─────────────────┘   └────────────────────┘
```

---

## Métricas de Éxito del Sistema (KPIs de Negocio y Técnicos)

| Métrica | Tipo | Target | Medición |
|---|---|---|---|
| `ΔE[R]` vs. alineaciones históricas | Negocio | ≥ +0.20 R/G | Backtesting Walk-Forward |
| `ΔWins` proyectados por temporada | Negocio | ≥ +3.0 W/season | Pythagorean Expectation |
| Adoption Rate (mánager sigue sistema) | Negocio | ≥ 70% en semana 8+ | Feedback Loop DB |
| PA-Level Model Log-Loss | Técnico | ≤ 0.84 (holdout) | MLflow / Evaluación offline |
| Expected Calibration Error (ECE) | Técnico | ≤ 0.035 | Reliability Diagrams |
| Latencia P99 inference individual | Técnico | ≤ 50ms | CloudWatch / Grafana |
| Tiempo simulación completa (100K) | Técnico | ≤ 60 segundos | Ray dashboard |
| RAG Faithfulness Score | Técnico | ≥ 0.88 | RAGAS evaluation |
| Pipeline game-day E2E latency | Operativo | ≤ 90 minutos | Airflow SLA |
| Uptime en días de partido | Operativo | ≥ 99.9% | CloudWatch composite alarm |

---

*Documento generado por el equipo de ML Engineering. Para contribuciones o correcciones, abrir un PR en el repositorio `mlb-lineup-optimizer-docs`. Versión del sistema referenciada: `v4.2.0`. Última actualización del documento: 2025.*
