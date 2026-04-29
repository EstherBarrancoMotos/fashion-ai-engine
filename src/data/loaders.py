"""
Optimized data loaders for the H&M dataset.

The transactions file has 31M rows and ~3 GB on disk. Naive `pd.read_csv`
takes minutes and uses huge RAM. These loaders use explicit dtypes and
optionally polars for speed.
"""

from __future__ import annotations

import pandas as pd

from src.config import ARTICLES_FILE, CUSTOMERS_FILE, TRANSACTIONS_FILE

# Explicit dtypes drastically reduce memory and speed up reading.
ARTICLES_DTYPES = {
    "article_id": "string",
    "product_code": "int32",
    "prod_name": "string",
    "product_type_no": "int16",
    "product_type_name": "category",
    "product_group_name": "category",
    "graphical_appearance_no": "int32",
    "graphical_appearance_name": "category",
    "colour_group_code": "int16",
    "colour_group_name": "category",
    "perceived_colour_value_id": "int8",
    "perceived_colour_value_name": "category",
    "perceived_colour_master_id": "int8",
    "perceived_colour_master_name": "category",
    "department_no": "int16",
    "department_name": "category",
    "index_code": "category",
    "index_name": "category",
    "index_group_no": "int8",
    "index_group_name": "category",
    "section_no": "int8",
    "section_name": "category",
    "garment_group_no": "int16",
    "garment_group_name": "category",
    "detail_desc": "string",
}

CUSTOMERS_DTYPES = {
    "customer_id": "string",
    "FN": "float32",
    "Active": "float32",
    "club_member_status": "category",
    "fashion_news_frequency": "category",
    "age": "float32",
    "postal_code": "string",
}

TRANSACTIONS_DTYPES = {
    "customer_id": "string",
    "article_id": "string",
    "price": "float32",
    "sales_channel_id": "int8",
}


def load_articles() -> pd.DataFrame:
    """Load the articles (product) catalogue."""
    return pd.read_csv(ARTICLES_FILE, dtype=ARTICLES_DTYPES)


def load_customers() -> pd.DataFrame:
    """Load the customer master table."""
    return pd.read_csv(CUSTOMERS_FILE, dtype=CUSTOMERS_DTYPES)


def load_transactions(nrows: int | None = None) -> pd.DataFrame:
    """
    Load the transactions table.

    Parameters
    ----------
    nrows : int | None
        If provided, only the first `nrows` are loaded — useful for
        prototyping. Pass None for the full ~31M rows.
    """
    df = pd.read_csv(
        TRANSACTIONS_FILE,
        dtype=TRANSACTIONS_DTYPES,
        parse_dates=["t_dat"],
        nrows=nrows,
    )
    return df


def load_transactions_polars(nrows: int | None = None):
    """
    Polars version — ~10x faster for full transactions file.
    Requires `pip install polars`.
    """
    import polars as pl

    df = pl.read_csv(
        TRANSACTIONS_FILE,
        n_rows=nrows,
        try_parse_dates=True,
    )
    return df
