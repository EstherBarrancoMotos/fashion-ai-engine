"""Entrenamiento del modelo final de demanda y persistencia en models/.

Reproduce la lógica del notebook 03b_Demand_Full_Dataset:
- Filtro cold-start (>= 4 semanas de histórico en train)
- Feature engineering temporal (lags + rolling means + calendario)
- Join con article_features (categóricas)
- Log-transform del target (units_sold) y de los lags
- Split temporal estricto (train hasta 2020-06-22, test desde 2020-08-22)
- LightGBM Regressor con categoricals nativos

Uso desde la raíz del proyecto:
    python -m src.training
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
MODELS = ROOT / "models"

RANDOM_STATE = 42
TRAIN_END = pd.Timestamp("2020-06-22")
VAL_END = pd.Timestamp("2020-08-22")

CATEGORY_COLS = [
    "product_group_name", "product_type_name", "colour_group_name",
    "department_name", "index_group_name", "garment_group_name",
]

NUM_FEATURES = [
    "log_lag_1", "log_lag_2", "log_lag_4",
    "log_rolling_mean_4", "log_rolling_mean_8",
    "month", "week_of_year", "avg_price",
]

FEATURE_COLS = NUM_FEATURES + CATEGORY_COLS
TARGET = "log_units"


def build_features(weekly: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    """Aplica filtro cold-start, feature engineering y join con metadata."""
    weekly = weekly.sort_values(["article_id", "week"]).reset_index(drop=True)

    # Filtro: productos con histórico suficiente en train
    weeks_in_train = (
        weekly[weekly["week"] <= TRAIN_END]
        .groupby("article_id", observed=True).size()
    )
    valid_articles = weeks_in_train[weeks_in_train >= 4].index
    weekly = weekly[weekly["article_id"].isin(valid_articles)].copy()

    # Lags y rolling means
    g = weekly.groupby("article_id", observed=True)["units_sold"]
    for lag in [1, 2, 4]:
        weekly[f"lag_{lag}"] = g.shift(lag)
    for window in [4, 8]:
        weekly[f"rolling_mean_{window}"] = (
            g.shift(1)
             .rolling(window=window, min_periods=1)
             .mean()
             .reset_index(level=0, drop=True)
        )

    # Calendario
    weekly["month"] = weekly["week"].dt.month
    weekly["week_of_year"] = weekly["week"].dt.isocalendar().week.astype(int)

    # Join con article_features
    weekly = weekly.merge(
        articles[["article_id"] + CATEGORY_COLS],
        on="article_id", how="left",
    )
    for col in CATEGORY_COLS:
        weekly[col] = weekly[col].astype("category")

    # Log-transform target y lags
    weekly["log_units"] = np.log1p(weekly["units_sold"])
    for col in ["lag_1", "lag_2", "lag_4", "rolling_mean_4", "rolling_mean_8"]:
        weekly[f"log_{col}"] = np.log1p(weekly[col])

    # Eliminar filas sin lags completos
    weekly = weekly.dropna(subset=["lag_4", "rolling_mean_8"]).copy()
    return weekly


def temporal_split(df: pd.DataFrame):
    train = df[df["week"] <= TRAIN_END].copy()
    val = df[(df["week"] > TRAIN_END) & (df["week"] <= VAL_END)].copy()
    test = df[df["week"] > VAL_END].copy()
    return train, val, test


def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("Cargando datasets procesados...")
    weekly = pd.read_parquet(PROCESSED / "weekly_sales.parquet")
    articles = pd.read_parquet(PROCESSED / "article_features.parquet")
    print(f"  weekly_sales : {weekly.shape}")
    print(f"  articles     : {articles.shape}")

    print("Feature engineering...")
    df = build_features(weekly, articles)
    print(f"  dataset modelado: {df.shape}")

    print("Split temporal...")
    train, val, test = temporal_split(df)
    train.to_parquet(TRAIN_DIR / "train.parquet", index=False)
    test.to_parquet(TEST_DIR / "test.parquet", index=False)
    print(f"  train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")

    print("Entrenando LightGBM con categoricals nativos...")
    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_val, y_val = val[FEATURE_COLS], val[TARGET]

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=CATEGORY_COLS)
    lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=CATEGORY_COLS,
                          reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": RANDOM_STATE,
    }

    model = lgb.train(
        params, lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    out = MODELS / "final_model.pkl"
    with open(out, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": FEATURE_COLS,
            "category_cols": CATEGORY_COLS,
            "best_iteration": model.best_iteration,
        }, f)
    print(f"Modelo guardado: {out}")
    print(f"  best_iteration: {model.best_iteration}")


if __name__ == "__main__":
    main()
