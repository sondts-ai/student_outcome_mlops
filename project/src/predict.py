import mlflow
import mlflow.sklearn
import pandas as pd

from project.src.data_preprocessing import CLASS_NAMES
from project.src.feature_engineering import add_engineered_features


MODEL_URI = "models:/bestmodel_v1@use_this_model"


def load_model():
    return mlflow.sklearn.load_model(MODEL_URI)

def load_model():
    return mlflow.sklearn.load_model(MODEL_URI)


def predict_sample(sample: dict):
    model = load_model()

    frame = pd.DataFrame([sample])
    frame = add_engineered_features(frame)

    prediction_index = int(model.predict(frame)[0])

    result = {
        "prediction": CLASS_NAMES[prediction_index]
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]

        result["probabilities"] = {
            class_name: float(probability)
            for class_name, probability in zip(
                CLASS_NAMES,
                probabilities,
            )
        }

    return result