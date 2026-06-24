"""DAG: pipeline de datos y reentrenamiento del MLB Lineup Optimizer.

Orquesta el pipeline batch en contenedores efímeros de la imagen ``mlb-pipeline``
(vía ``DockerOperator``). Ventajas de este diseño:

  - Airflow NO necesita las dependencias del proyecto → sin conflictos de versiones.
  - Cada tarea ejecuta un script del proyecto y escribe en los directorios montados
    del HOST (``data/``, ``models/``, ``results/``, ``reports/``), los MISMOS que lee
    el frontend/API en local. El frontend NO se modifica ni se conteneriza.

Flujo (semanal): ingesta Silver → Gold → entrenamiento (con gate de despliegue) →
backtest out-of-sample. Si el gate falla, ``train_v3.py`` devuelve código !=0 y la
tarea ``train_model`` se marca como fallida (el modelo NO se promociona).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# Ruta del proyecto en el HOST (inyectada por docker-compose desde .env).
HOST_PROJECT_DIR = os.environ["HOST_PROJECT_DIR"]
PIPELINE_IMAGE = os.environ.get("PIPELINE_IMAGE", "mlb-pipeline:latest")

# Un único bind-mount: el proyecto del host -> /app del contenedor del pipeline.
# Así los scripts ven el código y persisten datos/modelos/resultados en el host.
_MOUNTS = [Mount(source=HOST_PROJECT_DIR, target="/app", type="bind")]


def _pipeline_task(task_id: str, command: str, **kwargs) -> DockerOperator:
    """Crea una tarea que ejecuta un comando en la imagen del pipeline.

    Args:
        task_id: Identificador de la tarea en el DAG.
        command: Comando a ejecutar dentro del contenedor (p. ej. ``python ...``).
        **kwargs: Overrides adicionales para ``DockerOperator``.

    Returns:
        El ``DockerOperator`` configurado.
    """
    return DockerOperator(
        task_id=task_id,
        image=PIPELINE_IMAGE,
        command=command,
        mounts=_MOUNTS,
        working_dir="/app",
        auto_remove="force",
        mount_tmp_dir=False,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        environment={"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")},
        **kwargs,
    )


default_args = {
    "owner": "mlb",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="mlb_data_pipeline",
    description="Ingesta Silver -> Gold -> entrenamiento (gate) -> backtest",
    schedule="0 8 * * 1",          # cada lunes a las 08:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlb", "training", "pipeline"],
) as dag:

    # 1) Refresca la temporada en curso en la capa Silver (incremental, --force).
    ingest_silver = _pipeline_task(
        "ingest_silver",
        "python build_silver.py --years {{ logical_date.year }} --force",
    )

    # 2) Reconstruye el dataset Gold (features con shift(1) anti-leakage).
    build_gold = _pipeline_task(
        "build_gold",
        "python scripts/build_gold_v3.py",
    )

    # 3) Entrena el modelo. Split temporal estricto + gate de drift; si el gate
    #    falla, train_v3.py sale con código !=0 y esta tarea falla (no promociona).
    train_model = _pipeline_task(
        "train_model",
        "python train_v3.py",
    )

    # 4) Backtest out-of-sample a nivel juego (IC bootstrap, sim-vs-realidad).
    backtest = _pipeline_task(
        "backtest",
        "python backtest.py",
    )

    ingest_silver >> build_gold >> train_model >> backtest
