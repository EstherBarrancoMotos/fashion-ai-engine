"""Cached data loaders para todas las páginas de Streamlit."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'reports' / 'results'
FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures'


@st.cache_data
def load_pnl_summary() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / 'simulator_pnl_summary.parquet')


@st.cache_data
def load_sensitivity() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / 'simulator_sensitivity.parquet')


@st.cache_data
def load_per_article() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / 'simulator_per_article.parquet')


@st.cache_data
def load_top20() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / 'simulator_top20_reassign.parquet')


@st.cache_data
def load_summary_json() -> dict:
    with open(RESULTS_DIR / 'simulator_summary.json', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_demand_predictions() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / 'demand_predictions.parquet')


@st.cache_data
def load_return_predictions() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / 'return_predictions.parquet')


@st.cache_data
def load_article_features() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / 'article_features.parquet')
    df['article_id'] = df['article_id'].astype(str).str.zfill(10)
    return df


@st.cache_data
def load_weekly_sales() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / 'weekly_sales.parquet')


@st.cache_data
def load_per_article_enriched() -> pd.DataFrame:
    """Per-article del simulador + atributos de articles para análisis por categoría."""
    per_art = load_per_article()
    per_art['article_id'] = per_art['article_id'].astype(str).str.zfill(10)
    feats = load_article_features()
    return per_art.merge(feats, on='article_id', how='left')