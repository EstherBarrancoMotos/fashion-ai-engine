"""
Page 2 — Demand Explorer
Explora el forecast de demanda artículo por artículo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.data_loaders import (
    load_demand_predictions,
    load_article_features,
)
from utils.styling import inject_css, kpi_card, fmt_units, COLORS

st.set_page_config(
    page_title="Demand Explorer · Fashion AI",
    page_icon="📈",
    layout="wide",
)
inject_css()

# ============================================================
# DATA
# ============================================================
preds = load_demand_predictions().copy()
preds['article_id'] = preds['article_id'].astype(str).str.zfill(10)
feats = load_article_features()

# WAPE por artículo (para distribución global y selector)
@st.cache_data
def article_wape_table(_preds: pd.DataFrame) -> pd.DataFrame:
    grp = _preds.groupby('article_id').agg(
        units_total=('units_sold', 'sum'),
        pred_total=('y_pred', 'sum'),
        abs_err=('units_sold', lambda s: 0),  # placeholder, computed below
        n_weeks=('week', 'nunique'),
    )
    # Compute WAPE properly
    err = (_preds.assign(abs_err=(_preds['units_sold'] - _preds['y_pred']).abs())
                 .groupby('article_id')['abs_err'].sum())
    grp['abs_err'] = err
    grp['wape'] = grp['abs_err'] / grp['units_total'].replace(0, np.nan)
    grp = grp.reset_index().merge(
        feats[['article_id', 'product_type_name', 'colour_group_name', 'section_name']],
        on='article_id', how='left'
    )
    return grp

wape_tbl = article_wape_table(preds)

# ============================================================
# HEADER
# ============================================================
st.markdown("# 📈 Demand Explorer")
st.markdown(
    "Explora la calidad del forecast artículo por artículo. "
    "El modelo (LightGBM con log-target) predice unidades vendidas semanales "
    "con un **WAPE global de 0.53** (mejora 6.9% sobre el baseline naive)."
)
st.markdown("---")

# ============================================================
# SELECTOR
# ============================================================
left, right = st.columns([2, 1])

with left:
    # Top 200 artículos por volumen (los relevantes)
    top_articles = (
        wape_tbl.sort_values('units_total', ascending=False)
                .head(200)
    )
    options = top_articles.apply(
        lambda r: f"{r['article_id']} · {r['product_type_name']} · {r['colour_group_name']} "
                  f"({int(r['units_total'])} uds · WAPE {r['wape']:.2f})",
        axis=1
    ).tolist()
    article_map = dict(zip(options, top_articles['article_id'].tolist()))

    selected_label = st.selectbox(
        "Selecciona un artículo (top 200 por volumen)",
        options,
        index=0,
    )
    selected_id = article_map[selected_label]

with right:
    st.markdown("&nbsp;")
    st.markdown("&nbsp;")
    st.info(f"**Article ID:** `{selected_id}`")

# ============================================================
# ARTICLE DETAIL
# ============================================================
art_preds = preds[preds['article_id'] == selected_id].sort_values('week')
art_meta = feats[feats['article_id'] == selected_id].iloc[0] if (feats['article_id'] == selected_id).any() else None
art_wape = wape_tbl[wape_tbl['article_id'] == selected_id].iloc[0]

# KPIs
st.markdown("### 📊 Métricas del artículo")
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Ventas reales (total)", fmt_units(art_wape['units_total']) + " uds")
with c2:
    kpi_card("Predicción (total)", fmt_units(art_wape['pred_total']) + " uds")
with c3:
    delta_color = art_wape['wape'] <= 0.53  # mejor que media global
    kpi_card(
        "WAPE",
        f"{art_wape['wape']:.2f}",
        f"{'mejor' if delta_color else 'peor'} que la media (0.53)",
        delta_positive=delta_color,
    )
with c4:
    kpi_card("Semanas observadas", f"{int(art_wape['n_weeks'])}")

st.markdown("")

# Time series chart
st.markdown("### 📅 Real vs Predicción semanal")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=art_preds['week'], y=art_preds['units_sold'],
    name='Real', mode='lines+markers',
    line=dict(color=COLORS['primary'], width=3),
    marker=dict(size=7),
))
fig.add_trace(go.Scatter(
    x=art_preds['week'], y=art_preds['y_pred'],
    name='Predicción', mode='lines+markers',
    line=dict(color=COLORS['optimized'], width=3, dash='dash'),
    marker=dict(size=7),
))
fig.update_layout(
    height=420,
    margin=dict(t=20, b=40, l=20, r=20),
    plot_bgcolor='white',
    xaxis_title="Semana",
    yaxis_title="Unidades vendidas",
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    hovermode='x unified',
)
st.plotly_chart(fig, use_container_width=True)

# Article metadata
if art_meta is not None:
    st.markdown("### 🏷️ Metadatos del artículo")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"**Tipo de producto**  \n{art_meta['product_type_name']}")
    m2.markdown(f"**Color**  \n{art_meta['colour_group_name']}")
    m3.markdown(f"**Sección**  \n{art_meta['section_name']}")
    m4.markdown(f"**Departamento**  \n{art_meta['department_name']}")

st.markdown("---")

# ============================================================
# GLOBAL CONTEXT
# ============================================================
st.markdown("### 🌍 Distribución del error en el catálogo")
st.markdown(
    "Esta es la distribución de WAPE por artículo en todo el catálogo modelado. "
    "El modelo es más fiable en artículos con volumen estable; los outliers de WAPE alto "
    "suelen ser artículos con muy pocas ventas o muy estacionales."
)

# Filter sensible WAPE values for histogram
wape_clean = wape_tbl[(wape_tbl['units_total'] >= 50) & wape_tbl['wape'].notna() & (wape_tbl['wape'] <= 3)]

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=wape_clean['wape'],
    nbinsx=50,
    marker=dict(color=COLORS['optimized'], line=dict(color='white', width=1)),
))
fig.add_vline(
    x=0.53, line_dash='dash', line_color=COLORS['naive'],
    annotation_text="WAPE global = 0.53",
    annotation_position='top right',
)
fig.update_layout(
    height=380,
    margin=dict(t=20, b=40, l=20, r=20),
    plot_bgcolor='white',
    xaxis_title="WAPE por artículo",
    yaxis_title="Nº de artículos",
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Sobre {len(wape_clean):,} artículos con ≥50 uds vendidas. "
    f"Mediana WAPE = {wape_clean['wape'].median():.2f}. "
    f"Artículos con WAPE < 0.5: {(wape_clean['wape']<0.5).mean()*100:.0f}%."
)