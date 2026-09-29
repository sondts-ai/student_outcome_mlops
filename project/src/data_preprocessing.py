from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


RANDOM_STATE = 42

TARGET = "Target"

CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]


NUMERICAL_COLS = [
    "Age at enrollment",
    "Application order",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]


CATEGORICAL_COLS = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the raw dataset."""

    data = pd.read_csv(path)

    expected = set(
        NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET]
    )

    if set(data.columns) != expected:
        raise ValueError(
            "Dataset columns do not match the expected schema."
        )

    return data


def split_dataset(data: pd.DataFrame):
    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(data[TARGET])

    if list(label_encoder.classes_) != CLASS_NAMES:
        raise ValueError(
            f"Unexpected target classes: {list(label_encoder.classes_)}"
        )

    X = data.drop(columns=[TARGET])

    return train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=y,
    )

def make_preprocessor(selected_numeric_cols=None):
 
    selected_numeric_cols = (
        selected_numeric_cols
        if selected_numeric_cols is not None
        else NUMERICAL_COLS
    )

    return ColumnTransformer(
        [
            (
                "num",
                "passthrough",
                selected_numeric_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLS,
            ),
        ]
    )