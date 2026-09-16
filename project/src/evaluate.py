from pathlib import Path
import sys

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix




from data_preprocessing import CLASS_NAMES, load_dataset, split_dataset
from feature_engineering import add_engineered_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT.parent / "dataset (1).csv"

MODEL_PATH = PROJECT_ROOT.parent / "models" / "best_model.joblib"


def evaluate_model():
    data = load_dataset(DATA_PATH)
    _, X_test, _, y_test = split_dataset(data)
    X_test = add_engineered_features(X_test)
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4%}")
    print("\nClassification report:")
    print(
        classification_report(y_test, predictions, target_names=CLASS_NAMES, digits=4)
    )
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions))

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }


if __name__ == "__main__":
    evaluate_model()
