"""Evaluación del modelo final sobre data/test/.

Carga models/final_model.pkl (entrenado por src/training.py) y reporta:
- WAPE del modelo
- WAPE del baseline naive (lag_1)
- Mejora porcentual

Uso desde la raíz del proyecto:
    python -m src.evaluation
"""
from pathlib import Path
import pickle
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "test"
MODELS = ROOT / "models"
RESULTS = ROOT / "reports" / "results"


def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else float("nan")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Cargando test set y modelo...")
    test = pd.read_parquet(TEST_DIR / "test.parquet")

    with open(MODELS / "final_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    category_cols = bundle["category_cols"]

    # Asegurar dtype categórico (LightGBM lo necesita)
    for col in category_cols:
        if col in test.columns:
            test[col] = test[col].astype("category")

    X_test = test[feature_cols]
    y_test = test["units_sold"].values  # target real (escala original)

    print("Prediciendo (escala log) y deshaciendo log...")
    y_pred_log = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)

    # Baseline naive: predecir con lag_1
    y_baseline = test["lag_1"].fillna(test["lag_1"].median()).values

    metrics = {
        "wape_model": wape(y_test, y_pred),
        "wape_baseline_lag1": wape(y_test, y_baseline),
        "n_test": int(len(y_test)),
        "model": "LightGBM (log target, native categoricals)",
        "feature_cols": feature_cols,
    }
    metrics["improvement_pct"] = float(
        100 * (metrics["wape_baseline_lag1"] - metrics["wape_model"])
        / metrics["wape_baseline_lag1"]
    )

    print(json.dumps(
        {k: v for k, v in metrics.items() if k != "feature_cols"},
        indent=2,
    ))

    out = RESULTS / "evaluation_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Métricas guardadas: {out}")


if __name__ == "__main__":
    main()
