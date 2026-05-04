"""Demand Explorer — análisis del modelo de demanda con drilldown por producto.

  - métricas técnicas del modelo (WAPE, mejora vs baseline, n predicciones)
  - selección interactiva de un producto → forecast vs real a lo largo del tiempo
  - top productos donde más se equivoca el modelo (candidatos para v2)
  - distribución global del error
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.data_loaders import (
    load_demand_predictions, load_article_features, RESULTS_DIR,
)
from utils.styling import (
    inject_css, kpi_card, eyebrow, section_divider, caption,
    fmt_eur, fmt_units, plotly_layout, COLORS,
)

st.set_page_config(
    page_title="Demand Explorer · Fashion AI Engine",
    page_icon="◆",
    layout="wide",
)
inject_css()

# ============================================================
# SIDEBAR BRANDING
# ============================================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 0.5rem 0 1.5rem 0; border-bottom: 1px solid {COLORS['border']};
                    margin-bottom: 1rem;">
            <div style="font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
                        color: {COLORS['accent']}; text-transform: uppercase;
                        margin-bottom: 0.25rem;">
                Demand Explorer
            </div>
            <div style="font-size: 0.95rem; font-weight: 600; color: {COLORS['primary']};
                        line-height: 1.3;">
                Forecast vs<br>actual reality.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# DATA
# ============================================================
preds = load_demand_predictions().copy()
features = load_article_features().copy()

preds['article_id'] = preds['article_id'].astype(str).str.zfill(10)
preds['week'] = pd.to_datetime(preds['week'])
preds['error'] = preds['y_pred'] - preds['units_sold']
preds['abs_error'] = preds['error'].abs()

# Cargar métricas
metrics_path = RESULTS_DIR / 'demand_metrics.json'
with open(metrics_path, encoding='utf-8') as f:
    model_metrics = json.load(f)

# Métricas resumen
n_articles = preds['article_id'].nunique()
n_predictions = len(preds)
wape_global = preds['abs_error'].sum() / preds['units_sold'].abs().sum()

# Métricas del JSON
lgbm_metrics = next(
    (r for r in model_metrics['results'] if r['model'].startswith('LightGBM')),
    None
)

# ============================================================
# HEADER
# ============================================================
eyebrow("Demand Module")
st.markdown("# Forecast vs reality")
caption(
    "Análisis del modelo de predicción de demanda a nivel artículo × semana. "
    "Las métricas globales del modelo, el drilldown por producto individual y "
    "los casos donde el modelo más se equivoca — input para iterar a v2."
)

section_divider()

# ============================================================
# MODEL METRICS
# ============================================================
st.markdown("## Modelo de demanda")
caption(
    "LightGBM con log-target, validación temporal estricta, categoricals "
    "nativos. Entrenado sobre 1.59M filas de train, evaluado sobre 47K filas "
    "de test (semanas posteriores a 2020-08-22)."
)

c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    kpi_card(
        "WAPE (test)",
        f"{lgbm_metrics['wape']:.3f}" if lgbm_metrics else f"{wape_global:.3f}",
        "menor = mejor",
        delta_kind="neutral",
    )
with c2:
    kpi_card(
        "Mejora vs baseline",
        f"+{model_metrics['improvement_pct']:.1f}%",
        "vs lag_1 naive",
        delta_kind="positive",
    )
with c3:
    kpi_card(
        "Productos predichos",
        fmt_units(n_articles),
        "en periodo de evaluación",
        delta_kind="neutral",
    )
with c4:
    kpi_card(
        "Predicciones totales",
        fmt_units(n_predictions),
        "filas article × week",
        delta_kind="neutral",
    )

# ============================================================
# COMPARATIVA DE BASELINES
# ============================================================
section_divider()
st.markdown("## Comparativa con baselines")
caption(
    "El modelo se compara con dos baselines naive: predecir el valor de la "
    "semana anterior (lag_1) y predecir la media móvil de las 4 semanas previas."
)

baselines_df = pd.DataFrame(model_metrics['results'])

fig = go.Figure(go.Bar(
    x=baselines_df['model'],
    y=baselines_df['wape'],
    marker_color=[
        COLORS['naive'] if m.startswith('Baseline') else COLORS['accent']
        for m in baselines_df['model']
    ],
    text=[f"{w:.3f}" for w in baselines_df['wape']],
    textposition='outside',
    hovertemplate="<b>%{x}</b><br>WAPE: %{y:.4f}<extra></extra>",
))
fig.update_layout(
    yaxis_title="WAPE",
    showlegend=False,
)
fig.update_xaxes(tickangle=15)
st.plotly_chart(plotly_layout(fig, height=380), use_container_width=True)

# ============================================================
# DRILLDOWN POR PRODUCTO
# ============================================================
section_divider()
st.markdown("## Forecast vs actual por producto")
caption(
    "Selecciona un artículo del catálogo para ver la serie temporal de "
    "demanda real (línea oscura) frente a la predicción del modelo (verde)."
)

# Construir lista de productos con info para el selector
article_summary = preds.groupby('article_id').agg(
    n_weeks=('week', 'count'),
    total_sold=('units_sold', 'sum'),
    wape=('abs_error', lambda x: x.sum() / preds.loc[x.index, 'units_sold'].abs().sum()),
).reset_index()

# Top 200 con más volumen para no saturar el selector
top_articles = article_summary.nlargest(200, 'total_sold')
top_articles_with_meta = top_articles.merge(
    features[['article_id', 'product_type_name', 'colour_group_name']],
    on='article_id', how='left',
)
top_articles_with_meta['label'] = (
    top_articles_with_meta['article_id'].astype(str) + '  ·  '
    + top_articles_with_meta['product_type_name'].astype(str).replace('nan', '?') + '  ·  '
    + top_articles_with_meta['colour_group_name'].astype(str).replace('nan', '?')
)

selected_label = st.selectbox(
    "Producto (top 200 por volumen)",
    options=top_articles_with_meta['label'].tolist(),
    index=0,
)
selected_id = selected_label.split('  ·  ')[0]

product_data = preds[preds['article_id'] == selected_id].sort_values('week')

# Métricas del producto seleccionado
prod_wape = product_data['abs_error'].sum() / product_data['units_sold'].abs().sum() if product_data['units_sold'].sum() > 0 else 0
prod_total_sold = int(product_data['units_sold'].sum())
prod_total_pred = int(product_data['y_pred'].sum())

cm1, cm2, cm3 = st.columns(3, gap="medium")
with cm1:
    kpi_card("WAPE de este producto",
             f"{prod_wape:.3f}",
             f"{len(product_data)} semanas observadas",
             delta_kind="neutral")
with cm2:
    kpi_card("Vendido real",
             f"{prod_total_sold:,} uds",
             "suma del periodo",
             delta_kind="neutral")
with cm3:
    diff = prod_total_pred - prod_total_sold
    kpi_card("Predicho",
             f"{prod_total_pred:,} uds",
             f"{diff:+,} vs real",
             delta_kind="positive" if abs(diff) < prod_total_sold * 0.2 else "negative")

# Gráfico forecast vs actual
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=product_data['week'],
    y=product_data['units_sold'],
    mode='lines+markers',
    line=dict(color=COLORS['primary'], width=2),
    marker=dict(size=7),
    name='Real',
    hovertemplate="Real<br>%{x|%Y-%m-%d}<br>%{y} uds<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=product_data['week'],
    y=product_data['y_pred'],
    mode='lines+markers',
    line=dict(color=COLORS['accent'], width=2, dash='dash'),
    marker=dict(size=7),
    name='Predicción',
    hovertemplate="Predicho<br>%{x|%Y-%m-%d}<br>%{y:.1f} uds<extra></extra>",
))
fig.update_layout(
    xaxis_title="Semana",
    yaxis_title="Unidades vendidas",
)
st.plotly_chart(plotly_layout(fig, height=420), use_container_width=True)

# ============================================================
# DISTRIBUCIÓN DEL ERROR
# ============================================================
section_divider()
st.markdown("## Distribución del error global")
caption(
    "Histograma del error de predicción (predicho − real) sobre las "
    f"{n_predictions:,} predicciones. Centrado en cero = modelo no sesgado. "
    "Cola larga en alguno de los lados = modelo sub o sobre-estima."
)

# Clip para que la cola no aplaste el histograma
err_clipped = preds['error'].clip(lower=preds['error'].quantile(0.01),
                                   upper=preds['error'].quantile(0.99))

fig = go.Figure(go.Histogram(
    x=err_clipped,
    nbinsx=60,
    marker_color=COLORS['accent'],
    marker_line_color='white',
    marker_line_width=1,
))
fig.add_vline(x=0, line_dash='dash', line_color=COLORS['primary'], line_width=2)
fig.add_annotation(
    x=0, y=1, yref='paper', yanchor='bottom',
    text=f"  Bias: {preds['error'].mean():+.2f}",
    showarrow=False, font=dict(color=COLORS['muted'], size=11),
)

fig.update_layout(
    xaxis_title="Error (predicho − real, unidades)",
    yaxis_title="N predicciones",
    showlegend=False,
)
st.plotly_chart(plotly_layout(fig, height=380), use_container_width=True)

# ============================================================
# TOP PRODUCTOS DONDE MÁS SE EQUIVOCA
# ============================================================
section_divider()
st.markdown("## Top productos donde más se equivoca el modelo")
caption(
    "Los 20 artículos con mayor WAPE individual (mínimo 4 semanas observadas). "
    "Estos son los candidatos para análisis cualitativo: ¿son productos nuevos? "
    "¿muy estacionales? ¿outliers? Inputs valiosos para iterar a v2."
)

worst = article_summary[article_summary['n_weeks'] >= 4].nlargest(20, 'wape')
worst = worst.merge(
    features[['article_id', 'product_type_name', 'colour_group_name',
              'garment_group_name', 'department_name']],
    on='article_id', how='left',
)

display_worst = worst[[
    'article_id', 'product_type_name', 'colour_group_name', 'garment_group_name',
    'n_weeks', 'total_sold', 'wape',
]].copy()
display_worst.columns = [
    'Article ID', 'Product type', 'Color', 'Garment', 'N weeks', 'Vendido', 'WAPE',
]
display_worst['WAPE'] = display_worst['WAPE'].round(3)
display_worst['Vendido'] = display_worst['Vendido'].astype(int)

st.dataframe(display_worst, use_container_width=True, hide_index=True, height=420)

# ============================================================
# Footer
# ============================================================
st.markdown(
    f"""
    <div style='text-align:center; color:{COLORS["muted"]};
                font-size:0.8rem; margin-top:3rem; padding-top:1.5rem;
                border-top: 1px solid {COLORS["border"]};'>
        Demand Explorer · {n_articles:,} articles · {n_predictions:,} predictions
        · WAPE {wape_global:.3f}
    </div>
    """,
    unsafe_allow_html=True,
)
