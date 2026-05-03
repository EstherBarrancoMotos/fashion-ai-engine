"""Entrenamiento del modelo final y persistencia en models/.

Lee data/processed/weekly_sales.parquet, hace split temporal estricto
(últimas 8 semanas como test), entrena pipeline LightGBM y guarda
models/final_model.pkl.

Uso:
    python -m src.training
"""
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
MODELS = ROOT / "models"

TEST_WEEKS = 8

NUM_FEATURES = ["avg_price"]  # mínimas para que el script sea autocontenido
TARGET = "units_sold"


def temporal_split(df: pd.DataFrame):
    df = df.sort_values("week")
    cutoff = df["week"].drop_duplicates().sort_values().iloc[-TEST_WEEKS]
    train = df[df["week"] < cutoff].copy()
    test = df[df["week"] >= cutoff].copy()
    return train, test


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
            verbosity=-1,
        )),
    ])


def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("Cargando dataset procesado...")
    df = pd.read_parquet(PROCESSED / "weekly_sales.parquet")
    print(f"  shape: {df.shape}")

    print("Split temporal...")
    train, test = temporal_split(df)
    train.to_parquet(TRAIN_DIR / "train.parquet", index=False)
    test.to_parquet(TEST_DIR / "test.parquet", index=False)
    print(f"  train: {len(train):,}  test: {len(test):,}")

    print("Entrenando pipeline (StandardScaler + LightGBM)...")
    pipe = build_pipeline()
    X_train = train[NUM_FEATURES]
    y_train = np.log1p(train[TARGET])
    pipe.fit(X_train, y_train)

    out = MODELS / "final_model.pkl"
    with open(out, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Modelo guardado: {out}")


if __name__ == "__main__":
    main()