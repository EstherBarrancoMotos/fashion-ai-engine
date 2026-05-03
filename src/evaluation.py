"""Evaluación del modelo final sobre data/test/.

Carga models/final_model.pkl y reporta WAPE + comparación con baseline naive.

Uso:
    python -m src.evaluation
"""
from pathlib import Path
import pickle
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "test"
MODELS = ROOT / "models"
RESULTS = ROOT / "reports" / "results"

NUM_FEATURES = ["avg_price"]
TARGET = "units_sold"


def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Cargando test set y modelo...")
    test = pd.read_parquet(TEST_DIR / "test.parquet")
    with open(MODELS / "final_model.pkl", "rb") as f:
        pipe = pickle.load(f)

    X_test = test[NUM_FEATURES]
    y_test = test[TARGET].values

    print("Prediciendo...")
    y_pred_log = pipe.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)

    y_baseline = np.full_like(y_test, fill_value=np.median(y_test), dtype=float)

    metrics = {
        "wape_model": float(wape(y_test, y_pred)),
        "wape_baseline": float(wape(y_test, y_baseline)),
        "n_test": int(len(y_test)),
    }
    metrics["improvement_pct"] = float(
        100 * (metrics["wape_baseline"] - metrics["wape_model"])
        / metrics["wape_baseline"]
    )

    print(json.dumps(metrics, indent=2))
    out = RESULTS / "evaluation_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Métricas guardadas: {out}")


if __name__ == "__main__":
    main()
    