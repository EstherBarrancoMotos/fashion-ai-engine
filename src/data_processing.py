"""Pipeline de procesamiento de datos.

Lee CSVs originales de data/raw/ (H&M Kaggle), aplica limpieza y feature
engineering, y persiste el dataset de modelado en data/processed/.

Uso:
    python -m src.data_processing
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def load_raw():
    """Carga las tres tablas crudas de H&M."""
    tx = pd.read_csv(
        RAW / "transactions_train.csv",
        parse_dates=["t_dat"],
        dtype={"article_id": str},
    )
    articles = pd.read_csv(RAW / "articles.csv", dtype={"article_id": str})
    customers = pd.read_csv(RAW / "customers.csv")
    return tx, articles, customers


def build_weekly_sales(tx: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    """Agrega transacciones a granularidad article × week.

    Versión condensada del feature engineering del notebook 02_LimpiezaEDA.
    Para el detalle completo ver el notebook.
    """
    tx = tx.copy()
    tx["week"] = tx["t_dat"].dt.to_period("W").dt.start_time

    weekly = (
        tx.groupby(["article_id", "week"])
          .agg(units_sold=("t_dat", "count"),
               avg_price=("price", "mean"))
          .reset_index()
    )

    art_cols = [
        "article_id", "product_type_name", "product_group_name",
        "graphical_appearance_name", "colour_group_name",
        "perceived_colour_master_name", "department_name",
        "index_group_name", "section_name", "garment_group_name",
    ]
    weekly = weekly.merge(articles[art_cols], on="article_id", how="left")
    return weekly


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print("Cargando raw...")
    tx, articles, customers = load_raw()
    print(f"  transactions: {len(tx):,}")
    print(f"  articles    : {len(articles):,}")
    print(f"  customers   : {len(customers):,}")

    print("Construyendo weekly_sales...")
    weekly = build_weekly_sales(tx, articles)
    out = PROCESSED / "weekly_sales.parquet"
    weekly.to_parquet(out, index=False)
    print(f"Guardado: {out}  ({len(weekly):,} filas x {len(weekly.columns)} cols)")


if __name__ == "__main__":
    main()