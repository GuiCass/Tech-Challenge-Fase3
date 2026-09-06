"""DAG de treino/retreino do classificador de urgência de laudos médicos.

Duas tasks encadeadas:
  1. ingest_data — baixa/atualiza o dataset em data/ (scripts/download_data.py).
  2. train_and_save_model — treina o pipeline TF-IDF + LogisticRegression e
     salva o artefato em models/ (src/ml/train.py).

Reaproveita o código já usado localmente e em CI, para que o comportamento do
retreino orquestrado seja o mesmo do treino manual.
"""

import sys
from datetime import datetime
from pathlib import Path

# Garante que o projeto (src/, scripts/) é importável independentemente de
# onde a pasta dags/ está montada dentro do AIRFLOW_HOME.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator

from scripts.download_data import main as ingest_data
from src.ml.train import train as train_and_save_model

default_args = {
    "owner": "triagem-laudos",
    "retries": 1,
}

with DAG(
    dag_id="triagem_laudos_train_dag",
    description="Pipeline de treino/retreino do classificador de urgência de laudos médicos",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["triagem", "treino"],
) as dag:
    ingest_data_task = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    train_and_save_model_task = PythonOperator(
        task_id="train_and_save_model",
        python_callable=train_and_save_model,
    )

    ingest_data_task >> train_and_save_model_task
