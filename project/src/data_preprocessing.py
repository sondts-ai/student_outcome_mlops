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
    """Load the raw CSV using the notebook's schema."""
    data = pd.read_csv(path)
    expected = set(NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET])
    if set(data.columns) != expected:
        raise ValueError("Dataset columns do not match the notebook schema.")
    return data


def split_dataset(data: pd.DataFrame):
    """Reproduce the notebook's stratified 80/20 train/test split."""
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[TARGET])
    if list(label_encoder.classes_) != CLASS_NAMES:
        raise ValueError(f"Unexpected target classes: {list(label_encoder.classes_)}")
    X = data.drop(columns=[TARGET])
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def make_preprocessor(numerical_cols=None):
    """Build the final notebook preprocessor: numeric passthrough + categorical one-hot."""
    numerical_cols = numerical_cols or NUMERICAL_COLS
    return ColumnTransformer(
        [
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
