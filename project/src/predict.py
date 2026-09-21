import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd



from project.src.data_preprocessing import CLASS_NAMES
from project.src.feature_engineering import add_engineered_features
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "dataset.csv"

MODEL_PATH = PROJECT_ROOT.parent / "models" / "best_model.joblib"


def predict_sample(sample: dict):
    model = joblib.load(MODEL_PATH)
    frame = pd.DataFrame([sample])
    frame = add_engineered_features(frame)
    prediction_index = int(model.predict(frame)[0])
    result = {"prediction": CLASS_NAMES[prediction_index]}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        result["probabilities"] = {
            class_name: float(probability)
            for class_name, probability in zip(CLASS_NAMES, probabilities)
        }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predict student outcome from one JSON sample."
    )

    parser.add_argument(
        "--json",
        required=True,
        help="Path to a JSON object containing raw feature values.",
    )

    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as file:
        sample = json.load(file)

    print(
        json.dumps(
            predict_sample(sample),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
