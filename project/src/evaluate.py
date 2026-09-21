
from pathlib import Path
import json

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import mlflow
from project.src.data_preprocessing import CLASS_NAMES, load_dataset, split_dataset
from project.src.feature_engineering import add_engineered_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "dataset.csv"
MODEL_PATH = PROJECT_ROOT.parent / "models" / "best_model.joblib"
METRICS_PATH = PROJECT_ROOT.parent / "reports" / "metrics.json"
mlflow.set_experiment("MLflow Quickstart")

def evaluate_model():
    data = load_dataset(DATA_PATH)

    _, X_test, _, y_test = split_dataset(data)

    X_test = add_engineered_features(X_test)

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    for class_name in CLASS_NAMES:
        mlflow.log_metric(
            f"{class_name}_precision",
            report[class_name]["precision"],
        )
        mlflow.log_metric(
            f"{class_name}_recall",
            report[class_name]["recall"],
        )
        mlflow.log_metric(
            f"{class_name}_f1",
            report[class_name]["f1-score"],
        )

    mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])
    mlflow.log_metric("weighted_f1", report["weighted avg"]["f1-score"])
    mlflow.log_metric("accuracy", accuracy)
    matrix = confusion_matrix(y_test, predictions)

    print(f"Accuracy: {accuracy:.4%}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(matrix)

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    mlflow.log_artifact(str(METRICS_PATH))
    print(f"\nSaved metrics: {METRICS_PATH}")

    return metrics


if __name__ == "__main__":

    evaluate_model()

