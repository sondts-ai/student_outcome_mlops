import numpy as np
import pandas as pd

from project.src.data_preprocessing import NUMERICAL_COLS

ENGINEERED_COLS = [
    "1st_sem_approval_rate",
    "2nd_sem_approval_rate",
    "approval_rate_change",
    "approved_units_change",
    "grade_change",
    "total_approved_units",
    "total_without_evaluation",
]
ENGINEERED_NUMERICAL_COLS = NUMERICAL_COLS + ENGINEERED_COLS


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce every feature used by the final notebook model."""
    result = data.copy()
    result["1st_sem_approval_rate"] = result[
        "Curricular units 1st sem (approved)"
    ] / result["Curricular units 1st sem (evaluations)"].replace(0, np.nan)
    result["2nd_sem_approval_rate"] = result[
        "Curricular units 2nd sem (approved)"
    ] / result["Curricular units 2nd sem (evaluations)"].replace(0, np.nan)
    result["approval_rate_change"] = (
        result["2nd_sem_approval_rate"] - result["1st_sem_approval_rate"]
    )
    result["approved_units_change"] = (
        result["Curricular units 2nd sem (approved)"]
        - result["Curricular units 1st sem (approved)"]
    )
    result["grade_change"] = (
        result["Curricular units 2nd sem (grade)"]
        - result["Curricular units 1st sem (grade)"]
    )
    result["total_approved_units"] = (
        result["Curricular units 1st sem (approved)"]
        + result["Curricular units 2nd sem (approved)"]
    )
    result["total_without_evaluation"] = (
        result["Curricular units 1st sem (without evaluations)"]
        + result["Curricular units 2nd sem (without evaluations)"]
    )
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)
