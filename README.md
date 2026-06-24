# MLB Lineup Optimizer

Sistema de analítica de béisbol que predice, partido a partido, la distribución de
resultados de cada turno al bate (*plate appearance*), simula el juego por Monte Carlo y
**optimiza el orden de bateo** para maximizar las carreras esperadas. Incluye un dashboard
web (React + FastAPI), explicaciones con LLM (RAG/War Room) y un conjunto de herramientas de
backtest y diagnóstico.

> **Qué funciona y qué no (validado empíricamente, ver [`reports/diagnostics/`](reports/diagnostics/)):**
> - ✅ **Optimización de lineup**: el orden óptimo aporta **+0.052 runs/juego** sobre el orden usado
>   (IC90 [+0.033, +0.072], estadísticamente significativo) ≈ **+8.5 runs/temporada**. Es el
>   propósito principal del sistema y cumple.
> - ✅ **Modelo PA**: calibración impecable (ECE 0.002 out-of-sample).
> - ⚠️ **Predicción del ganador (win-probability)**: ≈ moneda. No por bugs, sino porque el
>   resultado de un partido de MLB es intrínsecamente casi aleatorio (incluso un Elo apenas supera
>   al 50/50). Úsese como contexto, no como ventaja de apuesta.

---

## Índice
1. [Cómo funciona](#cómo-funciona)
2. [Arquitectura](#arquitectura)
3. [Estructura del repositorio](#estructura-del-repositorio)
4. [Instalación](#instalación)
5. [Flujo de datos y entrenamiento](#flujo-de-datos-y-entrenamiento)
6. [Uso](#uso)
7. [Automatización (Docker + Apache Airflow)](#automatización-docker--apache-airflow)
8. [Evaluación y diagnóstico](#evaluación-y-diagnóstico)
9. [Contratos e invariantes](#contratos-e-invariantes)
10. [Tests](#tests)
11. [Limitaciones](#limitaciones)

---

## Cómo funciona

El pipeline va de datos crudos de Statcast a una recomendación de alineación:

```
Statcast (pybaseball / MLB StatsAPI)
        │  build_silver.py
        ▼
data/silver/plate_appearances/   ← un registro por turno al bate (PA)
        │  scripts/build_gold_v3.py   (features con shift(1) anti-leakage)
        ▼
data/gold/features_train_v3.parquet   ← ~2.06M PA, 12 temporadas, 51 features
        │  train_v3.py   (LightGBM 8 clases + calibración isotónica + gate de drift)
        ▼
models/at_bat_predictor.pkl   ← predice P(out, K, BB, 1B, 2B, 3B, HR, DP) por PA
        │
        ├─ predict_tonight.py ─► para cada bateador: vector de 8 probabilidades
        │        ▼
        │   src/simulation/  ← Monte Carlo (cadena de Markov base-outs, 100k juegos)
        │        ▼            E[R], P(W), percentiles
        │   src/optimizer/   ← algoritmo genético: busca el orden que maximiza E[R]
        │
        └─ Servido por api/main.py (FastAPI) ◄── frontend/ (React SPA)
```

**Modelo (`AtBatPredictor`)**: LightGBM multiclase de 8 outcomes + `CalibratedClassifierCV`
isotónico. Las probabilidades alimentan el simulador como pesos estocásticos, por lo que la
**calibración** (no el rebalanceo de clases) es el objetivo.

**Simulador (`MonteCarloEngine`)**: cadena de Markov de 24 estados base-outs con tablas de
avance de corredores probabilísticas, park factors, extra innings (ghost runner) y, opcionalmente,
relevo del bullpen rival. Devuelve E[R], P(W) y percentiles con su error de muestreo.

**Optimizador (`GeneticLineupOptimizer`)**: búsqueda genética de 3 capas (seeding sabermétrico →
GA → refinamiento de los top-K con 100k simulaciones) sobre las permutaciones del orden de bateo,
con test de significancia del ganador vs el segundo.

---

## Arquitectura

| Componente | Ruta | Rol |
|---|---|---|
| **Features (contrato único)** | `src/features/shared_features.py` | Definición ÚNICA de features; la usan training y serving (anti-skew) |
| **Constantes** | `src/constants.py` | Fuente única: shrinkage James-Stein, run values, park factors, prior de liga |
| **Modelo** | `src/models/model_at_bat.py` | `AtBatPredictor` (LightGBM + isotónica + ECE) |
| **Simulación** | `src/simulation/simulation_engine.py` | Monte Carlo base-outs (Numba/Ray) |
| **Optimizador** | `src/optimizer/lineup_optimizer.py` | GA de orden de bateo |
| **RAG / LLM** | `src/rag/` | Scouting + explicación de alineación (Anthropic Claude) |
| **Backend SPA** | `api/main.py` | FastAPI (puerto 8000) que sirve al frontend React |
| **Microservicio GA** | `app/main.py` | FastAPI separado: predicción PA individual + optimización async |
| **Frontend** | `frontend/` | SPA React + Vite (dnd-kit, etc.) |
| **CLI predicción** | `predict_tonight.py` | Predice/optimiza partidos del día desde la terminal |
| **Rutina diaria** | `morning.py` | post-game + schedule + predicciones |

---

## Estructura del repositorio

```
.
├── predict_tonight.py        # CLI principal de predicción/optimización
├── morning.py                # rutina diaria
├── build_silver.py           # ingesta Statcast → capa Silver (PA-level)
├── train_v3.py               # entrenamiento del modelo (Gold → modelo + gate)
├── backtest.py               # backtest out-of-sample a nivel juego (IC bootstrap)
├── docker/                   # Dockerfile.pipeline (runner batch) + Dockerfile.airflow
├── docker-compose.airflow.yml  # stack de automatización (Airflow + Postgres)
├── dags/                     # DAGs de Airflow (pipeline semanal + predicciones diarias)
├── requirements-airflow.txt  # deps de la imagen del pipeline (sin servidor web)
├── scripts/
│   ├── build_gold_v3.py      # Silver → Gold (features de entrenamiento)
│   ├── promote_model.py      # promoción champion/challenger
│   ├── diagnose_pipeline.py  # descomposición del error (PA-OOS / win-prob)
│   ├── ceiling_test.py       # techo de discriminación del modelo
│   ├── feature_screen.py     # screening de features nuevas por AUC OOS
│   └── backtest_lineups.py   # valor contrafactual del optimizador
├── src/                      # lógica de negocio (features, models, simulation, optimizer, rag)
├── api/                      # backend FastAPI del dashboard (+ capas económica, fatiga, shadow)
├── app/                      # microservicio FastAPI de predicción/optimización
├── frontend/                 # SPA React + Vite
├── data/
│   ├── raw/                  # parquets Statcast crudos
│   ├── silver/plate_appearances/season=YYYY/   # PA-level por temporada
│   └── gold/features_train_v3.parquet          # dataset de entrenamiento
├── models/                   # at_bat_predictor.pkl (modelo en producción)
├── results/<fecha>/<ABBR>.json    # predicciones generadas
├── reports/                  # backtest, diagnostics, auditoría
├── tests/                    # pytest (incluye guardián de paridad de features)
└── requirements*.txt         # dependencias (ver más abajo)
```

> `build_gold.py` (v2) está **deprecado**; el Gold activo es `features_train_v3.parquet`
> generado por `scripts/build_gold_v3.py`.

---

## Instalación

**Requisitos**: Python **3.11 o 3.12** (NO 3.13+), y Node.js 18+ para el frontend. El repo usa
**Git LFS** para los parquets/modelos — instala `git lfs` antes de clonar.

### Backend / pipeline (Python)

```bash
# 1. Crear y activar el entorno virtual
python -m venv .venv
.venv\Scripts\activate           # Windows (PowerShell/Git Bash)
# source .venv/bin/activate      # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements_local.txt    # uso local (CLI + dashboard); recomendado
# pip install -r requirements_api.txt    # sólo el servicio API
# pip install -r requirements.txt        # set completo
```

### Frontend (React)

```bash
cd frontend
npm install
npm run dev        # servidor de desarrollo Vite
# npm run build    # build de producción
```

### Variables de entorno

- `ANTHROPIC_API_KEY` — necesaria para las explicaciones LLM (War Room / RAG scouting).
- El resto de fuentes (MLB StatsAPI, Statcast vía `pybaseball`) son públicas y no requieren clave.

---

## Flujo de datos y entrenamiento

Ejecutar en orden (solo cuando quieras regenerar datos o reentrenar):

```bash
# 1. Ingesta Statcast → Silver (PA-level). --years es una LISTA de años (por defecto 2015-2020).
python build_silver.py --years 2021 2022 2023 2024 2025 2026
python build_silver.py --years 2026 --force      # refrescar la temporada en curso

# 2. Silver → Gold (features con shift(1) anti-leakage)
python scripts/build_gold_v3.py

# 3. Entrenar el modelo. Split temporal estricto:
#    train < VAL_SEASON (= año actual − 1)  <  calibración/holdout (= VAL_SEASON)
#    Aplica el GATE de despliegue (drift por clase + sesgo E[R/PA] ≤ 0.005);
#    si falla, NO copia el modelo a producción.
python train_v3.py

# 4. (Opcional) Promoción champion/challenger con holdout virgen
python scripts/promote_model.py --challenger-pkl models/pa_predictor_v1.pkl --dry-run
```

Salida: `models/at_bat_predictor.pkl` (modelo servido) si pasa el gate.

---

## Uso

### Predicción / optimización por terminal

```bash
# Lista los partidos de hoy y deja elegir
python predict_tonight.py

# Un equipo concreto (busca su partido del día)
python predict_tonight.py --team NYY
python predict_tonight.py --team LAD --side home

# Otra fecha
python predict_tonight.py --date 2026-05-20

# TODOS los partidos del día, guardando JSON por equipo en results/<fecha>/
python predict_tonight.py --all --output-dir results

# Optimizar el orden de bateo (algoritmo genético) y/o ajustar nº de simulaciones
python predict_tonight.py --team NYY --optimize --n-sims 100000

# Saltar la simulación (solo features/probabilidades)
python predict_tonight.py --team NYY --no-sim
```

Cada predicción guardada incluye un flag **`sim_status`**:
- `two_sided` — win-probability de una simulación consistente de dos lados (válida).
- `vs_league_avg` — sólo un lineup disponible; win-prob vs rival promedio (no comparable).
- `no_sim` — sin simulación.

### Rutina diaria

```bash
python morning.py                 # post-game de ayer + schedule + predicciones de hoy
python morning.py --date 2026-06-24
python morning.py --no-predict    # sólo post-game/schedule
```

### Dashboard web (backend + frontend)

```bash
# Terminal 1 — backend FastAPI (puerto 8000), sirve al SPA
python -m api.main
# o:  uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd frontend && npm run dev
```

Endpoints principales del backend (`api/main.py`, prefijo `/v1`): `games/today`,
`optimize/{game_pk}`, `report/{game_pk}`, `metrics/rolling`, `metrics/calibration`,
`metrics/backtest`, `track-record`, `economic/ev`, `economic/clv`, `fatigue/{player_id}`.
Operación: `GET /health`, `GET /metrics` (Prometheus).

### Microservicio de inferencia/optimización (`app/main.py`)

Servicio FastAPI separado, orientado a integración programática:
`POST /v1/predict/at-bat`, `POST /v1/optimize/lineup` (async), `GET /health`, `GET /metrics`.

---

## Automatización (Docker + Apache Airflow)

El proyecto incluye scaffolding **opcional** para ejecutar el pipeline de datos/modelo
de forma automática y programada. **No es necesario para el uso normal** (todo funciona en
local), pero está presente por si alguien quiere operarlo desatendido.

> **El frontend NO cambia.** La automatización sólo orquesta el pipeline *batch*. Airflow
> escribe en `data/`, `models/`, `results/` y `reports/` del host (vía volúmenes), y tu
> frontend + `api/main.py` siguen ejecutándose **en local** leyendo esos mismos directorios.

### Diseño

```
┌─────────────── Docker ───────────────┐         Host (local)
│  Apache Airflow (LocalExecutor)       │   ┌────────────────────────┐
│  scheduler + webserver + PostgreSQL   │   │  data/  models/        │
│            │ DockerOperator           │   │  results/  reports/    │ ◄─┐
│            ▼                          │   └────────────────────────┘   │ escribe
│  Contenedor efímero `mlb-pipeline`    │──────────► (volúmenes montados) ┘
│  (ejecuta build_silver / build_gold / │
│   train_v3 / morning / backtest)      │   ┌────────────────────────┐
└───────────────────────────────────────┘   │  frontend + api/main.py│ ◄── leen los
                                             │  (se ejecutan EN LOCAL)│     mismos dirs
                                             └────────────────────────┘
```

Airflow no lleva las dependencias del proyecto: lanza la imagen `mlb-pipeline` por cada
tarea (`DockerOperator`), evitando conflictos de versiones.

### Puesta en marcha

Requisitos: **Docker** y **Docker Compose** (Docker Desktop en Windows/Mac).

```bash
# 1. Configurar el entorno: copia la plantilla y edita HOST_PROJECT_DIR (ruta ABSOLUTA del proyecto)
cp .env.airflow.example .env
#   HOST_PROJECT_DIR=/ruta/absoluta/al/proyecto   (Windows: C:\Users\ferna\Desktop\MLB AI)

# 2. Construir la imagen del pipeline (contiene el código + deps batch)
docker build -f docker/Dockerfile.pipeline -t mlb-pipeline:latest .

# 3. Levantar el stack de Airflow (construye su imagen, inicia Postgres + scheduler + webserver)
docker compose -f docker-compose.airflow.yml up -d

# 4. Abrir la UI en http://localhost:8080  (usuario/clave del .env, por defecto airflow/airflow)
#    Activar (unpause) los DAGs deseados.

# Parar / limpiar:
docker compose -f docker-compose.airflow.yml down            # parar
docker compose -f docker-compose.airflow.yml down -v         # parar y borrar la BBDD de Airflow
```

### DAGs incluidos (`dags/`)

| DAG | Schedule (por defecto) | Qué hace |
|---|---|---|
| `mlb_data_pipeline` | semanal (lun 08:00) | `build_silver` → `build_gold_v3` → `train_v3` (con gate) → `backtest` |
| `mlb_daily_predictions` | diario (13:00 UTC) | `morning.py` (post-game + schedule + predicciones del día) |

Si el **gate de despliegue** del entrenamiento falla, `train_v3.py` termina con código ≠0 y la
tarea `train_model` se marca como fallida: el modelo **no** se promociona a producción.

### Notas

- **Sin conflictos de dependencias**: el pipeline corre en su propia imagen; Airflow sólo añade
  `apache-airflow-providers-docker`.
- **DockerOperator** necesita el socket de Docker (`/var/run/docker.sock`), ya montado en el
  compose. En Docker Desktop (Windows/Mac) funciona sin pasos extra.
- Ajusta los `schedule` de los DAGs a tu zona horaria / calendario MLB.
- Ejecutar un paso suelto sin Airflow (la misma imagen):
  `docker run --rm -v "$PWD":/app mlb-pipeline:latest python morning.py`

---

## Evaluación y diagnóstico

```bash
# Backtest a nivel juego (out-of-sample) con IC bootstrap, sim-vs-realidad y forma del win-prob.
# Sólo puntúa predicciones VÁLIDAS de dos lados (excluye las degeneradas).
python backtest.py                                   # toda la ventana disponible
python backtest.py --from 2026-05-01 --out reports/backtest/backtest.json

# Descomposición del error: ¿PA model, simulador u opponent? (PA-OOS 2026 + dispersión win-prob)
python scripts/diagnose_pipeline.py

# ¿La baja discriminación es del método o intrínseca? (oráculo con calidad de contacto)
python scripts/ceiling_test.py

# ¿Una feature nueva sube el AUC OOS antes de integrarla en el contrato anti-skew?
python scripts/feature_screen.py

# Valor real del optimizador: E[R] óptimo vs usado vs aleatorio, con IC y significancia
python scripts/backtest_lineups.py --max-games 40
```

Los hallazgos por fase están documentados en `reports/diagnostics/`
(`CHECKPOINT_0.md`, `FASE1_FINDINGS.md`, `FASE2_FINDINGS.md`, `FASE3_FINDINGS.md`).

---

## Contratos e invariantes

Reglas que el código mantiene y que **no deben romperse** (validadas en auditorías previas):

- **Contrato anti-skew**: `src/features/shared_features.py` es la única definición de features.
  Training (`scripts/build_gold_v3.py`) y serving (`predict_tonight.py`) la comparten; el serving
  no reimplementa features. Guardián: `tests/test_feature_parity.py`.
- **Sin class weights**: las probabilidades alimentan el simulador como pesos estocásticos; deben
  quedar **calibradas**, no rebalanceadas. No reintroducir escalares post-hoc tipo `_MC_RUNS_SCALE`
  (= 1.0 es el sentinel de "modelo calibrado").
- **Sin leakage intra-PA**: `xwoba`, `launch_speed/angle`, `pitch_count_in_pa`, `last_pitch_type`
  sólo se conocen al terminar el PA → excluidos del modelo (`_LEAKING`).
- **Split temporal estricto** + **gate de despliegue** en `train_v3.py`: si el drift por clase o el
  sesgo E[R/PA] superan el umbral, el modelo no se promociona.
- **Park factors / `is_home`** son **features** del modelo; el motor MC recibe factores neutrales
  cuando el modelo ya los incorpora (no aplicarlos dos veces).

---

## Tests

```bash
pytest                       # toda la suite
pytest tests/test_feature_parity.py     # contrato anti-skew train/serve
pytest tests/test_simulation_engine.py tests/test_lineup_optimizer.py
```

---

## Limitaciones

- **Predicción del ganador ≈ moneda**: techo intrínseco del béisbol; el modelo no añade ventaja
  medible sobre el 50/50 a nivel partido. El valor está en E[R] relativo / lineup, no en el binario.
- **Techo de datos del PA model**: con los datos actuales la discriminación está saturada
  (ver `reports/diagnostics/FASE1_FINDINGS.md`). Subirla requeriría **nueva fuente de datos**
  (Statcast "stuff" completo: velocidad/spin/arsenal, o weather histórico).
- **Modelado de bullpen** en el optimizador es parcial (roadmap); el GA usa el perfil del abridor.
- **LFS**: los artefactos grandes (parquets, modelos) viven en Git LFS; clona con `git lfs` instalado.

---

*Proyecto personal de analítica MLB. Datos vía MLB StatsAPI / Statcast (pybaseball); explicaciones
con Claude (Anthropic).*
