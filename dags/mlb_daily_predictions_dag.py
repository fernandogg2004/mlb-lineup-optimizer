"""DAG: rutina diaria de predicciones del MLB Lineup Optimizer.

Ejecuta ``morning.py`` (post-game de ayer + schedule del día + predicciones) en un
contenedor efímero de la imagen ``mlb-pipeline``. Los resultados se escriben en
``results/<fecha>/`` del HOST, que el frontend/API local leen tal cual. Aquí NO se
modifica el frontend ni su forma de obtener los datos.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

HOST_PROJECT_DIR = os.environ["HOST_PROJECT_DIR"]
PIPELINE_IMAGE = os.environ.get("PIPELINE_IMAGE", "mlb-pipeline:latest")
_MOUNTS = [Mount(source=HOST_PROJECT_DIR, target="/app", type="bind")]

default_args = {
    "owner": "mlb",
    "retries": 2,
    "retry_delay": timedelta(minutes=15),
}

with DAG(
    dag_id="mlb_daily_predictions",
    description="Rutina diaria: post-game + schedule + predicciones (morning.py)",
    schedule="0 13 * * *",          # cada día 13:00 UTC (ajusta a tu zona / horario MLB)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlb", "daily", "predictions"],
) as dag:

    daily_routine = DockerOperator(
        task_id="morning_routine",
        image=PIPELINE_IMAGE,
        command="python morning.py",
        mounts=_MOUNTS,
        working_dir="/app",
        auto_remove="force",
        mount_tmp_dir=False,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        environment={"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")},
    )
