from project.src.evaluate import evaluate_model
from project.src.train import train_model
import mlflow

if __name__ == "__main__":
    mlflow.set_experiment("MLflow Quickstart")

    with mlflow.start_run():
        train_model()
        evaluate_model()