from pathlib import Path
import sys

import joblib
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import mlflow
from sklearn.metrics import accuracy_score

from data_preprocessing import (
    RANDOM_STATE,
    load_dataset,
    make_preprocessor,
    split_dataset,
)
from feature_engineering import ENGINEERED_NUMERICAL_COLS, add_engineered_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "dataset.csv"

MODEL_PATH = PROJECT_ROOT.parent / "models" / "best_model.joblib"

mlflow.set_experiment("MLflow Quickstart")


def build_final_model():
    """Create the unchanged RandomOverSampler + tuned Random Forest pipeline."""
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(ENGINEERED_NUMERICAL_COLS)),
            ("oversampler", RandomOverSampler(random_state=RANDOM_STATE)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=700,
                    criterion="entropy",
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=2,
                    max_features=0.5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model():
    data = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = split_dataset(data)
    X_train = add_engineered_features(X_train)
    model = build_final_model()

    params={ "n_estimators": 700, "criterion": "entropy", "max_depth": "None",
     "min_samples_split": 2, "min_samples_leaf": 2, "max_features": 0.5, "class_weight": "balanced", }
    mlflow.log_params(params)

    #train
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Training samples: {len(X_train)}")
    print(f"Held-out test samples: {len(X_test)}")

    #log model to flow
    mlflow.sklearn.log_model( model, name="student-mlops", 
                            skops_trusted_types=[
                            "imblearn.over_sampling._random_over_sampler.RandomOverSampler",
                            "imblearn.pipeline.Pipeline",
                            "sklearn.tree._tree.Tree",
                        ],)
    return model


if __name__ == "__main__":

    train_model()