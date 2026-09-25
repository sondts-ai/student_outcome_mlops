from datetime import datetime
import sys

sys.path.insert(0, "/mnt/d/do_an_mon_hoc/he_quyet_dinh")

from airflow import DAG
from airflow.operators.python import PythonOperator

from project.src.validate_data import validate_data
from project.src.train import train_model
from project.src.evaluate import evaluate_model
from project.src.quality_gate import quality_gate

with DAG(
    dag_id="student_mlops",
    start_date=datetime(2026, 9, 21),
    schedule=None,
    catchup=False,
) as dag:

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=train_model,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=evaluate_model,
    )

    quality_task = PythonOperator(
    task_id="quality_gate",
    python_callable=quality_gate,
    )

    validate >> train >> evaluate>>quality_task