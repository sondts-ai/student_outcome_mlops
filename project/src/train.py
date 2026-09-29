from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from project.src.data_preprocessing import (
    RANDOM_STATE,
    load_dataset,
    make_preprocessor,
    split_dataset,
)

from project.src.feature_engineering import (
    BEST_CV_NUMERIC_COLS,
    add_engineered_features,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "dataset.csv"

MODEL_PATH = PROJECT_ROOT.parent / "models" / "best_model.joblib"


mlflow.set_experiment("Student Dropout Prediction")


def build_final_model():
    """Create the final Random Forest pipeline."""
    return Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(BEST_CV_NUMERIC_COLS),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    class_weight=None,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model():
    data = load_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test = split_dataset(data)

    # Feature engineering
    X_train = add_engineered_features(X_train)

    # Build final model
    model = build_final_model()

    params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
        "feature_set": "engineered_top12",
    }

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)

        # Save model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print(f"Saved model: {MODEL_PATH}")
        print(f"Training samples: {len(X_train)}")
        print(f"Held-out test samples: {len(X_test)}")

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            name="student-mlops",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
            ],
        )

    return None


if __name__ == "__main__":
    train_model()