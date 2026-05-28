# Modelo de Predicción de Carreras en Béisbol

## Visión General

Este sistema predice el **número esperado de carreras** (`E[R]`) que anotará cada equipo en un partido de béisbol. La predicción no se obtiene ajustando una regresión directa sobre carreras por partido, sino a través de una **cadena de dos modelos** encadenados:

```
Datos históricos (2015–presente)
        │
        ▼
[Modelo de turno al bate (PA-level)]
 Predice la distribución de probabilidad
 sobre 7 resultados posibles por turno
        │
        ▼
[Motor de Simulación Monte Carlo]
 Simula 100.000 partidos completos usando
 las probabilidades del modelo anterior
        │
        ▼
 E[R], P(W), percentiles, intervalos
```

Este diseño descompone el problema en su unidad atómica (un turno al bate) y recombina los resultados mediante física de béisbol exacta.

---

## Fase 1 — Ingesta de Datos

### Fuente

| Fuente | Mecanismo | Período |
|--------|-----------|---------|
| **Statcast (MLB)** vía `pybaseball` | Descarga histórica por temporada (batch) | 2015 – presente |
| **MLB StatsAPI live feed** | Polling en tiempo real cada 30 segundos | Día de partido |

### Capa Bronze (datos crudos)

Los datos crudos se escriben en **Delta Lake** particionado por `game_date`. Para cada lanzamiento de cada turno al bate se registran, entre otros campos:

- `release_speed` — velocidad de salida del lanzamiento (mph)
- `release_spin_rate` — tasa de giro (rpm)
- `pfx_x`, `pfx_z` — quiebre horizontal e inducido vertical (pulgadas)
- `plate_x`, `plate_z` — coordenadas de cruce sobre el plato
- `launch_speed` — velocidad de salida del batazo (mph)
- `launch_angle` — ángulo de salida del batazo (grados)
- `hit_distance_sc` — distancia recorrida del batazo (pies)
- `events` — resultado del turno (strikeout, single, home_run, …)
- `stand` / `p_throws` — lateralidad del bateador y lanzador

---

## Fase 2 — Limpieza de Datos y Control de Calidad

### Filtro de límites físicos (Bronze → Bronze limpio)

Antes de pasar a la capa Silver se eliminan filas con lecturas de sensores imposibles. Los umbrales son:

| Variable | Mínimo | Máximo | Justificación |
|---|---|---|---|
| `release_speed` | 40 mph | 110 mph | Rango físico de cualquier lanzamiento MLB |
| `release_spin_rate` | 0 rpm | 4 000 rpm | Techo teórico para cualquier tipo de lanzamiento |
| `launch_speed` | 0 mph | 135 mph | Máximo histórico registrado: 122.4 mph (G. Stanton) + margen sensor |

Las filas fuera de rango se **cuentan y emiten como métrica** (no se silencian), lo que activa alertas si la tasa de descarte supera un umbral operacional.

### Validación con Great Expectations (Silver)

La tabla Silver `plate_appearances` se valida con tres niveles de severidad:

| Nivel | Comportamiento | Ejemplos de reglas |
|---|---|---|
| **CRITICAL** | El lote va a cuarentena; pipeline se detiene | `launch_speed` ∉ [0, 135]; `launch_angle` ∉ [−90°, 90°] |
| **WARNING** | Se marca `_ge_validated = "WARN"` pero avanza | Temperaturas extremas de clima |
| **INFORMATIONAL** | Solo se registra para monitoreo de deriva | Distribuciones de estadísticas de bateo |

---

## Fase 3 — Feature Engineering

El pipeline de features produce un **vector de ~224 dimensiones** para cada turno al bate (PA). Se construye desde cuatro módulos independientes.

### 3.1 Features Rolling del Bateador (`features_rolling.py`)

Se calculan ventanas temporales de rendimiento reciente del bateador **sin fuga de datos**: para un partido en la fecha `D`, solo se usan datos de `[D−W días, D−1]`.

La garantía anti-fuga se implementa con:
1. `shift(1)` antes de cada operación rolling (excluye el partido corriente).
2. `join_asof` con `strategy="backward"` para ventanas de días calendario exactos.

#### Variables generadas (~20 dimensiones)

| Variable | Descripción | Ventanas |
|---|---|---|
| `xwoba_{7d/15d/30d}` | Promedio móvil de *expected wOBA* | 7, 15, 30 días |
| `launch_speed_{7d/15d/30d}` | Velocidad media de salida del batazo | 7, 15, 30 días |
| `xwoba_ewma_alpha02` | Media exponencialmente ponderada (α=0.2, estable) | Toda la historia |
| `xwoba_ewma_alpha05` | Media exponencialmente ponderada (α=0.5, reactiva) | Toda la historia |
| `k_rate_{7d/30d}` | Tasa de ponches | 7, 30 días |
| `bb_rate_{7d/30d}` | Tasa de bases por bolas | 7, 30 días |
| `hr_rate_30d` | Tasa de jonrones | 30 días |
| `hard_hit_rate_30d` | Tasa de batazos duros (≥ 95 mph EV) | 30 días |
| `babip_30d` | *BABIP* móvil | 30 días |
| `pa_{7d/15d/30d}` | Conteo de turnos al bate (indicador de actividad) | 7, 15, 30 días |

**Umbral mínimo de PAs:** ventana corta (7d) requiere ≥ 3 PAs; ventana larga (30d) requiere ≥ 10 PAs. Si no se llega al umbral, la métrica es `null` (se imputa a 0 posteriormente).

### 3.2 Splits por Platón con Bayesianismo Empírico (`features_platoon.py`)

Se calculan las estadísticas del bateador **segmentadas por la lateralidad del lanzador** (zurdo vs. diestro) y también **por tipo de lanzamiento** (FF, SL, CH, CU, etc.).

El problema central es que los splits por platón tienen **alta varianza muestral** para bateadores con pocos turnos contra una mano específica. Se resuelve con *James-Stein Empirical Bayes Shrinkage* (Efron & Morris, 1975; Tango-Lichtman-Dolphin, "The Book" 2007):

$$\hat{\theta}_{\text{estabilizado}} = \mu_{\text{prior}} + (1 - B) \times (\hat{\theta}_{\text{raw}} - \mu_{\text{prior}})$$

donde el coeficiente de contracción `B` es:

$$B = \frac{T}{T + PA_{\text{observados}}}$$

`T` es el **umbral de estabilización** (PAs necesarios para que la estadística sea 50% señal / 50% ruido):

| Estadística | T (PAs) | Fuente |
|---|---|---|
| wOBA | 200 | Lichtman (2010) |
| K% | 60 | Lichtman (2010) |
| BB% | 120 | Lichtman (2010) |
| BABIP | 500 | Lichtman (2010) |
| ISO | 160 | Lichtman (2010) |

**Interpretación:** Un bateador zurdo con solo 40 turnos contra zurdos y wOBA observado de 0.500 se contrae fuertemente hacia la media de la liga (~0.315). Un veterano con 400 turnos es casi plenamente confiado. Esto evita que el modelo sobreoptimice matchups ruidosos.

#### Métricas calculadas (~28 dimensiones)

Por cada combinación (bateador, lateralidad lanzador) y por tipo de lanzamiento:

- `woba_stabilized` — wOBA estabilizado por Bayes Empírico
- `k_rate_stabilized`, `bb_rate_stabilized`
- `iso_stabilized` — Poder Aislado estabilizado
- `babip_stabilized`
- `woba_shrinkage_b` — coeficiente de contracción (diagnóstico)

### 3.3 Embeddings del Arsenal del Lanzador (`player_embeddings.py`)

Se entrena un **autoencoder simétrico** en PyTorch que comprime el arsenal completo de un lanzador (hasta 9 tipos de lanzamiento × 7 atributos = 63 dimensiones) en un vector latente de **16 dimensiones**.

#### Atributos por tipo de lanzamiento (63 dims total)

| Atributo | Descripción | Normalización |
|---|---|---|
| `usage_pct` | Porcentaje de uso del lanzamiento | [0, 1] |
| `avg_velocity` | Velocidad promedio (mph) | [40, 110] → [0, 1] |
| `avg_spin_rate` | Tasa de giro promedio (rpm) | [800, 3800] → [0, 1] |
| `avg_pfx_x` | Quiebre horizontal promedio (pulgadas) | [−25, 25] → [0, 1] |
| `avg_pfx_z` | Quiebre vertical inducido promedio (pulgadas) | [−25, 25] → [0, 1] |
| `whiff_rate` | Tasa de swinging strikes en este lanzamiento | [0, 1] |
| `put_away_rate` | Tasa de ponches en conteos de 2 strikes | [0, 1] |

#### Arquitectura del Autoencoder

```
Encoder:   63 → 128 → 64 → 16   [BatchNorm + ReLU + Dropout(0.2)]
Decoder:   16 → 64 → 128 → 63   [BatchNorm + ReLU + Sigmoid]
```

- **Loss:** MSE (reconstrucción)  
- **Optimizador:** AdamW (lr=1e-3, weight_decay=1e-4)  
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=8 épocas)  
- **Early stopping:** patience=15 épocas  
- **Inicialización de pesos:** Kaiming uniform (apropiado para ReLU)

En la inferencia se descarta el decoder; solo se despliega el encoder.

**Ventaja sobre PCA:** El autoencoder captura relaciones no lineales en el espacio de arsenales (e.g., la interacción física entre velocidad × giro × quiebre es multiplicativa, no lineal). En el espacio latente, lanzadores con arsenales similares quedan próximos en distancia euclidiana, lo que permite transferencia de estadísticas a rookies sin historial (se usa el vecino más cercano en el espacio latente).

Se generan también embeddings de **16 dimensiones para el bateador** siguiendo la misma arquitectura, totalizando **32 dims adicionales** al vector de features.

### 3.4 Variables de Contexto y Parque

Incluidas en el vector final (~15 dimensiones):

| Variable | Descripción |
|---|---|
| `inning` | Número de entrada |
| `outs` | Outs en la entrada |
| `bases_state` | Estado de las bases (3-bit bitmask, 8 posibles estados) |
| `score_diff` | Diferencia de carreras al momento |
| `temp_f`, `humidity_pct` | Temperatura y humedad |
| `wind_u`, `wind_v` | Componentes del viento |
| `pressure_mb` | Presión atmosférica |
| `park_factor_hr` | Factor de parque para jonrones (e.g., Coors Field ≈ 1.35) |
| `park_factor_xb` | Factor de parque para extra-bases |
| `sin_day_of_year`, `cos_day_of_year` | Codificación cíclica del día (estacionalidad) |
| `sin_season_phase`, `cos_season_phase` | Fase de la temporada (cíclica) |

### Vector Final de Features

| Grupo | Dimensiones aprox. |
|---|---|
| Rolling bateador (7/15/30d + EWMA) | ~20 |
| Platoon splits estabilizados | ~28 |
| Embedding bateador (latente) | 16 |
| Embedding lanzador (latente) | 16 |
| Arsenal lanzador por tipo de lanzamiento | ~63 |
| Clima + parque + contexto + codificaciones cíclicas | ~15 |
| **Total** | **~158–224** |

---

## Fase 4 — Modelo de Turno al Bate (`model_at_bat.py`)

### Objetivo del modelo

El modelo **no predice carreras directamente**. Predice la **distribución de probabilidad sobre 7 resultados** posibles para cada turno al bate:

| Clase | Índice | Descripción |
|---|---|---|
| `OUT_IN_PLAY` | 0 | Out en juego (roletazo, elevado, línea) |
| `STRIKEOUT` | 1 | Ponche |
| `WALK_HBP` | 2 | Base por bolas o golpe de lanzamiento |
| `SINGLE` | 3 | Sencillo |
| `DOUBLE` | 4 | Doble |
| `TRIPLE` | 5 | Triple |
| `HOME_RUN` | 6 | Jonrón |

El vector de salida **p ∈ ℝ⁷** es una distribución de probabilidad (suma = 1.0) para el par bateador × lanzador × contexto.

### Algoritmo: LightGBM Multicalse + Calibración Isotónica

#### Hiperparámetros del modelo base

| Parámetro | Valor | Descripción |
|---|---|---|
| `objective` | multiclass | 7 clases |
| `n_estimators` | 1 500 | Rondas de boosting |
| `learning_rate` | 0.03 | Tasa de aprendizaje |
| `num_leaves` | 127 | Máximo de hojas (controla complejidad) |
| `min_child_samples` | 50 | Mínimo de datos en una hoja (regularización) |
| `subsample` | 0.80 | Sub-muestreo de filas por árbol |
| `colsample_bytree` | 0.75 | Sub-muestreo de features por árbol |
| `reg_alpha` | 0.10 | Regularización L1 |
| `reg_lambda` | 1.50 | Regularización L2 |
| `class_weight` | "balanced" | Rebalancea clases minoritarias (HR, Triple son raros) |
| `early_stopping_rounds` | 75 | Parada temprana si val-loss no mejora |

#### Calibración Post-hoc

Las probabilidades brutas de GBDT son conocidamente **sobre-confiadas** (over-confident en la clase mayoritaria). Dado que el motor de simulación usa estas probabilidades como pesos estocásticos, la mala calibración se propaga directamente al sesgo en `E[R]`.

Se aplica calibración isotónica mediante `CalibratedClassifierCV(cv='prefit', method='isotonic')`:

$$P_{\text{calibrado}}(c) = f_{\text{isotónica}}(P_{\text{raw}}(c))$$

La función `f_isotónica` es no paramétrica y consistente, lo que la hace superior al escalado de Platt (sigmoid) para el caso multiclase.

**Criterio de aceptación:** Error de Calibración Esperado (**ECE**) ≤ 0.035 en el conjunto de validación.

$$\text{ECE}_c = \sum_{b} \frac{|B_b|}{N} \cdot \left| \text{accuracy}(B_b) - \text{confidence}(B_b) \right|$$

$$\text{ECE} = \frac{1}{C} \sum_{c=1}^{C} \text{ECE}_c$$

Si el ECE supera 0.050 tras calibración isotónica, se emite una advertencia (el operador decide si cambiar a Platt o reajustar).

### Validación Temporal (Walk-Forward Cross-Validation)

Se usa `TimeSeriesSplit` con **5 pliegues temporales** que respetan el orden causal: el modelo **nunca se entrena con datos de temporadas que se están evaluando**. Esta es la única estrategia válida para series temporales deportivas.

Por cada pliegue:
- Primera mitad del conjunto de validación: calibración isotónica
- Segunda mitad: evaluación del ECE

### Métricas Registradas en MLflow

| Métrica | Descripción |
|---|---|
| `raw_logloss` | Log-loss del modelo sin calibrar |
| `cal_logloss` | Log-loss del modelo calibrado |
| `raw_ece` | ECE antes de calibración |
| `cal_ece` | ECE tras calibración |
| `ece_{clase}` | ECE por cada una de las 7 clases |
| `ece_reduction_pct` | % de reducción de ECE gracias a calibración |
| `avg_expected_run_value_per_pa` | Valor de carrera esperado por PA (sanity check) |

El **valor de carrera esperado por PA** es un sanity check que usa los pesos lineales de FanGraphs 2024:

| Resultado | Valor en carreras |
|---|---|
| OUT_IN_PLAY | 0.00 |
| STRIKEOUT | 0.00 |
| WALK_HBP | 0.33 |
| SINGLE | 0.47 |
| DOUBLE | 0.77 |
| TRIPLE | 1.04 |
| HOME_RUN | 1.40 |

$$E[R_{\text{PA}}] = \mathbf{p} \cdot \mathbf{w}$$

Si este valor diverge de ~0.12–0.16 (rango histórico de la liga), el modelo tiene un error.

### Prevención de Data Leakage

Se excluyen explícitamente del vector de features las métricas derivadas del propio PA objetivo:
```python
_LEAKING = {"xwoba", "launch_speed", "launch_angle"}
```
Estas son consecuencias del resultado que se predice, no predictores.

---

## Fase 5 — Motor de Simulación Monte Carlo (`simulation_engine.py`)

### Marco Matemático: Cadena de Markov

Un semi-inning de béisbol es una **cadena de Markov** con **24 estados**:

$$\text{estado} = (\text{outs} \in \{0,1,2\}) \times (\text{bases} \in \{0…7\}) = 3 \times 8 = 24 \text{ estados}$$

Las bases se codifican como un entero de 3 bits:
- bit 0 (valor 1) → corredor en 1ª base
- bit 1 (valor 2) → corredor en 2ª base
- bit 2 (valor 4) → corredor en 3ª base

El estado terminal (absorbente) es `outs == 3`.

### Tablas de Transición Precomputadas

Para cada par `(resultado, estado_bases)` se precomputan tablas deterministicas `O(1)`:

- `NEW_BASES_TABLE[resultado, bases]` → nuevo estado de bases
- `RUNS_TABLE[resultado, bases]` → carreras anotadas en el turno

Las reglas de avance de corredores codificadas son:
- **Jonrón:** todos los corredores y el bateador anotan; bases quedan vacías
- **Triple:** todos los corredores anotan; bateador en 3ª
- **Doble:** corredor en 2ª y 3ª anotan; corredor en 1ª pasa a 3ª; bateador en 2ª
- **Sencillo:** corredor en 3ª anota; 2ª→3ª; 1ª→2ª; bateador en 1ª
- **Base por bolas / HBP:** avance forzado; bateador a 1ª; solo se fuerza a los que corresponde
- **Out / Ponche:** bases sin cambio; outs +1

### Proceso de Simulación

Para cada simulación de partido:
1. Se alternan semi-innings de equipo A (ofensiva) y equipo B (ofensiva)
2. En cada PA se **muestrea un resultado** usando el método de CDF inverso:

$$u \sim \text{Uniform}[0,1], \quad \text{resultado} = \min\{i : \sum_{j=0}^{i} p_j \geq u\}$$

3. Se aplica la corrección por **factor de parque** antes del muestreo:

$$p'_{\text{HR}} = p_{\text{HR}} \times f_{\text{HR}}^{\text{parque}}, \quad p'_{\text{2B/3B}} = p_{\text{2B/3B}} \times f_{\text{XB}}^{\text{parque}}$$

El vector se renormaliza para mantener suma = 1.0.

4. El orden de bateo se **preserva entre entradas** (la rotación continúa desde el último bateador de la entrada anterior, igual que en béisbol real).

### Ejecución Paralela (dos capas)

| Capa | Tecnología | Alcance |
|---|---|---|
| **Capa 1** | Numba JIT + `prange` (OpenMP) | Paralelismo de hilos en una máquina, ~0.5–2 μs por partido |
| **Capa 2** | Ray remote actors | Paralelismo entre máquinas; 100k partidos / 20 workers = 5k cada uno |

**Target de rendimiento:** 100 000 simulaciones en < 10 segundos en una máquina de 16 cores.

### Resultados Extraídos

Después de N = 100 000 simulaciones:

| Métrica | Descripción |
|---|---|
| `E[R]` | Carreras esperadas anotadas (media sobre N simulaciones) |
| `E[RA]` | Carreras esperadas permitidas |
| `P(W)` | Probabilidad de victoria (con distribución 50/50 de empates) |
| `run_diff_mean` | Diferencial promedio de carreras |
| `runs_scored_percentiles` | Percentiles {5, 25, 50, 75, 95} de carreras anotadas |
| `shutout_pct` | Fracción de simulaciones en que el equipo rival anota 0 |
| `close_game_win_pct` | Win% en partidos con margen ≤ 1 carrera |
| `win_prob_ci_low/high` | Intervalo de confianza 90% para P(W) |
| `std_dev_runs_scored` | Desviación estándar de carreras anotadas |
| `percentile_10`, `percentile_90` | Intervalo de predicción (ancho objetivo: 15–25 pp) |

### Validación: Expectativa Pitagórica

Se calcula como cross-check independiente de `P(W)`:

$$P(W)_{\text{Pitágoras}} = \frac{E[R]^2}{E[R]^2 + E[RA]^2}$$

Si diverge significativamente de la `win_probability` simulada, indica un bug en la simulación.

---

## Fase 6 — Optimización del Lineup (GA)

El motor de simulación es el **evaluador de fitness** de un Algoritmo Genético que busca el lineup de 9 bateadores (orden de bateo) que maximiza `E[R]`. El GA usa el modo rápido del motor (5 000 simulaciones en lugar de 100 000) para reducir la latencia por evaluación.

---

## Conclusiones Matemáticas

### Por qué este diseño funciona

1. **Modularidad estadística**: El clasificador LightGBM aprende de millones de turnos al bate individuales (granularidad mayor), lo cual es más eficiente que tratar de ajustar directamente carreras por partido (~162 obs./temporada por equipo).

2. **La calibración isotónica es crítica**: Las probabilidades mal calibradas del GBDT base se propagan directamente como sesgo sistemático en `E[R]` a través del simulador. El objetivo ECE ≤ 0.035 está específicamente elegido para mantener ese sesgo por debajo de 0.1 carreras por partido.

3. **Empirical Bayes en los splits**: Sin contracción, matchups con pocos PAs (e.g., un bateador zurdo con solo 20 turnos contra zurdos) generarían features extremas que el modelo malinterpretaría. La contracción hacia la media poblacional es la solución óptima bayesiana cuando PA → 0.

4. **La cadena de Markov es el modelo de física correcto**: El béisbol tiene memoria (estado de bases y outs), pero dentro de un semi-inning la transición siguiente solo depende del estado actual (no de cómo se llegó ahí). La cadena de Markov con 24 estados es la representación matemáticamente exacta de esta estructura.

5. **Walk-forward CV evita data snooping**: Un modelo deportivo entrenado y evaluado con k-fold estándar (aleatorio) sufre contaminación temporal severa. La estrategia de división temporal garantiza que las métricas reportadas reflejen rendimiento real en predicción de partidos futuros.

### Limitaciones conocidas

| Limitación | Impacto | Mitigación implementada |
|---|---|---|
| El simulador juega exactamente 9 entradas (sin extras) | ~5–8% de los partidos terminan empatados tras 9 entradas | Los empates se distribuyen 50/50 entre win_prob y opp_win_prob |
| La alineación del orden de bateo no modela cambios de lanzadores mid-game | Sesgo moderado en entradas tardías | Factor de parque + calibración ECE absorben parte del efecto |
| Statcast existe solo desde 2015 | No hay datos anteriores de alta resolución | Pool de ~10 temporadas × ~185k PAs/temporada = muestra suficiente |
| Jugadores de primer año sin historial propio | Features rolling serán 0 (imputación por nulidad) | Embedding de vecino más cercano en el espacio latente del autoencoder |
