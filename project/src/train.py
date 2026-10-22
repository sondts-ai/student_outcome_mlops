
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from project.src.data_preprocessing import (
    RANDOM_STATE,
    load_dataset,
    make_preprocessor,
    split_dataset,
)

from project.src.feature_engineering import (
    MODEL_NUMERIC_COLS,
    add_engineered_features,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "dataset.csv"
MODEL_PATH = PROJECT_ROOT.parent / "models" / "best_model.joblib"

mlflow.set_experiment("Student Dropout Prediction")


def build_final_model():
    return Pipeline([
        (
            "preprocessor",
            make_preprocessor(MODEL_NUMERIC_COLS),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


def train_model():
    # 1. Load and split data
    data = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = split_dataset(data)

    # 2. Feature engineering
    X_train_fe = add_engineered_features(X_train)
    X_test_fe = add_engineered_features(X_test)

    # 3. Inspect missing values before training
    train_nan_count = X_train_fe.isna().sum().sum()
    test_nan_count = X_test_fe.isna().sum().sum()

    print(f"NaN in training features: {train_nan_count}")
    print(f"NaN in test features: {test_nan_count}")

    if train_nan_count > 0:
        print(
            "Columns containing NaN in training data:\n",
            X_train_fe.isna().sum()[
                X_train_fe.isna().sum() > 0
            ],
        )

    # Replace infinities with NaN in numerical columns.
    # The preprocessor must impute these missing values.
    X_train_fe = X_train_fe.replace([np.inf, -np.inf], np.nan)
    X_test_fe = X_test_fe.replace([np.inf, -np.inf], np.nan)

    # 4. Build and train model
    model = build_final_model()

    params = {
        "n_estimators": 500,
        "max_depth": "None",
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "feature_set": "engineered_top12",
        "random_state": RANDOM_STATE,
    }

    with mlflow.start_run():
        mlflow.log_params(params)

        model.fit(X_train_fe, y_train)

        # 5. Evaluate held-out test set
        y_pred = model.predict(X_test_fe)

        test_metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_macro_precision": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "test_macro_recall": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "test_macro_f1": f1_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "test_weighted_f1": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
        }

        mlflow.log_metrics(test_metrics)

        # 6. Save model locally
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print(f"Saved model: {MODEL_PATH}")
        print(f"Training samples: {len(X_train_fe)}")
        print(f"Held-out test samples: {len(X_test_fe)}")
        print("\nTest metrics:")

        for name, value in test_metrics.items():
            print(f"{name}: {value:.4f}")

        # 7. Log model artifact to MLflow
        mlflow.sklearn.log_model(
            model,
            name="student_mlops",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

    return test_metrics


if __name__ == "__main__":
    train_model()