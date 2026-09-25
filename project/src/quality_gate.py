
import json
from pathlib import Path


METRICS_PATH = Path("reports/metrics.json")
MIN_F1 = 0.70


def quality_gate():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_PATH}"
        )

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    f1 = metrics["classification_report"]["macro avg"]["f1-score"]

    print(f"F1 macro: {f1:.4f}")
    print(f"Required F1: {MIN_F1:.2f}")

    if f1 < MIN_F1:
        raise ValueError(
            f"Model quality is too low: "
            f"F1={f1:.4f} < {MIN_F1:.2f}"
        )

    print("Quality gate passed!")


if __name__ == "__main__":
    quality_gate()

