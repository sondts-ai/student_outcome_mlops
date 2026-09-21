import sys
from datetime import datetime

sys.path.insert(0, "/mnt/d/do_an_mon_hoc/he_quyet_dinh")

from airflow import DAG
from airflow.operators.python import PythonOperator

from project.src.train import train_model
from project.src.evaluate import evaluate_model


with DAG(
    dag_id="student_mlops",
    start_date=datetime(2026, 9, 21),
    schedule=None,
    catchup=False,
) as dag:

    train = PythonOperator(
        task_id="train",
        python_callable=train_model,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=evaluate_model,
    )

    train >> evaluate