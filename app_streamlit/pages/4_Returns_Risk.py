"""Returns Risk — métricas del modelo de devoluciones + análisis a nivel artículo.

Complementa Style Intelligence (que mira categorías) con análisis individual:
  - métricas técnicas del modelo (ROC-AUC, PR-AUC, Brier, calibración)
  - distribución global de probabilidades
  - top 50 productos de mayor riesgo (tabla descargable)
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
    load_return_predictions, load_article_features, RESULTS_DIR,
)
from utils.styling import (
    inject_css, kpi_card, eyebrow, section_divider, caption,
    fmt_eur, fmt_units, plotly_layout, COLORS,
)

st.set_page_config(
    page_title="Returns Risk · Fashion AI Engine",
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
                Returns Risk
            </div>
            <div style="font-size: 0.95rem; font-weight: 600; color: {COLORS['primary']};
                        line-height: 1.3;">
                Per-article risk<br>profile.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# DATA
# ============================================================
predictions = load_return_predictions().copy()
features = load_article_features().copy()

predictions['article_id'] = predictions['article_id'].astype(str).str.zfill(10)

df = predictions.merge(features, on='article_id', how='left')

# Cargar métricas del modelo
metrics_path = RESULTS_DIR / 'returns_metrics.json'
with open(metrics_path, encoding='utf-8') as f:
    model_metrics = json.load(f)

# ============================================================
# HEADER
# ============================================================
eyebrow("Risk Module")
st.markdown("# Returns risk profile")
caption(
    "Análisis individual del riesgo de devolución por artículo. Las métricas "
    "del modelo, la distribución de probabilidades predichas y el listado de "
    "los productos con mayor riesgo — accionable para decisiones de producción."
)

section_divider()

# ============================================================
# MODEL METRICS
# ============================================================
st.markdown("## Modelo de devoluciones")
caption(
    "LightGBM classifier sobre labels semi-sintéticas derivadas de tasas "
    "sectoriales (Narvar 2023, Optoro 2023). Validación temporal estricta."
)

c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    kpi_card(
        "ROC-AUC",
        f"{model_metrics['roc_auc']:.3f}",
        "discriminación general",
        delta_kind="neutral",
    )
with c2:
    kpi_card(
        "PR-AUC",
        f"{model_metrics['pr_auc']:.3f}",
        "calidad en clase positiva",
        delta_kind="neutral",
    )
with c3:
    kpi_card(
        "Brier score",
        f"{model_metrics['brier_score']:.3f}",
        "calibración (menor = mejor)",
        delta_kind="neutral",
    )
with c4:
    kpi_card(
        "Tasa real positivos",
        f"{model_metrics['positive_rate']*100:.1f}%",
        "base rate del problema",
        delta_kind="neutral",
    )

# ============================================================
# DISTRIBUCIÓN DE PROBABILIDADES
# ============================================================
section_divider()
st.markdown("## Distribución del riesgo en el catálogo")
caption(
    "Histograma de la probabilidad de devolución predicha sobre los "
    f"{len(df):,} artículos. Las zonas sombreadas marcan los umbrales de "
    "riesgo bajo (<15%), medio (15-30%) y alto (>30%)."
)

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=df['avg_return_prob'] * 100,
    nbinsx=40,
    marker_color=COLORS['accent'],
    marker_line_color='white',
    marker_line_width=1,
    hovertemplate="%{x:.1f}% · %{y} productos<extra></extra>",
))
# Vlines para umbrales
fig.add_vline(x=15, line_dash='dash', line_color=COLORS['muted'])
fig.add_vline(x=30, line_dash='dash', line_color=COLORS['bad'])
# Anotaciones
fig.add_annotation(x=7.5, y=0, yref='paper', yanchor='top',
                   text="Bajo", showarrow=False,
                   font=dict(color=COLORS['accent'], size=11))
fig.add_annotation(x=22.5, y=0, yref='paper', yanchor='top',
                   text="Medio", showarrow=False,
                   font=dict(color='#f59e0b', size=11))
fig.add_annotation(x=45, y=0, yref='paper', yanchor='top',
                   text="Alto", showarrow=False,
                   font=dict(color=COLORS['bad'], size=11))

fig.update_layout(
    xaxis_title="Probabilidad de devolución predicha (%)",
    yaxis_title="N productos",
    showlegend=False,
)
st.plotly_chart(plotly_layout(fig, height=400), use_container_width=True)

# Stats de la distribución
low = (df['avg_return_prob'] < 0.15).sum()
mid = ((df['avg_return_prob'] >= 0.15) & (df['avg_return_prob'] < 0.30)).sum()
high = (df['avg_return_prob'] >= 0.30).sum()

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    kpi_card("Riesgo bajo (<15%)", fmt_units(low),
             f"{low/len(df)*100:.1f}% del catálogo", delta_kind="positive")
with c2:
    kpi_card("Riesgo medio (15-30%)", fmt_units(mid),
             f"{mid/len(df)*100:.1f}% del catálogo", delta_kind="neutral")
with c3:
    kpi_card("Riesgo alto (>30%)", fmt_units(high),
             f"{high/len(df)*100:.1f}% del catálogo", delta_kind="negative")

# ============================================================
# CALIBRACIÓN: predicho vs real
# ============================================================
section_divider()
st.markdown("## Calibración del modelo")
caption(
    "Comparación entre la probabilidad predicha (avg_return_prob) y la tasa "
    "real observada (actual_return_rate) por bins de probabilidad. Una "
    "calibración perfecta seguiría la diagonal verde."
)

# Filtrar artículos con suficientes transacciones
df_cal = df[df['n_transactions'] >= 5].copy()

# Bin por probabilidad predicha
bins = np.linspace(0, df_cal['avg_return_prob'].quantile(0.99), 11)
df_cal['bin'] = pd.cut(df_cal['avg_return_prob'], bins=bins, include_lowest=True)

calibration = df_cal.groupby('bin', observed=True).agg(
    pred_mean=('avg_return_prob', 'mean'),
    actual_mean=('actual_return_rate', 'mean'),
    n=('article_id', 'count'),
).reset_index()
calibration = calibration[calibration['n'] >= 10]

fig = go.Figure()
# Diagonal perfecta
fig.add_trace(go.Scatter(
    x=[0, calibration['pred_mean'].max()],
    y=[0, calibration['pred_mean'].max()],
    mode='lines',
    line=dict(color=COLORS['accent'], dash='dash', width=2),
    name='Calibración perfecta',
    hoverinfo='skip',
))
# Curva real
fig.add_trace(go.Scatter(
    x=calibration['pred_mean'],
    y=calibration['actual_mean'],
    mode='lines+markers',
    line=dict(color=COLORS['primary'], width=3),
    marker=dict(size=10, color=COLORS['primary']),
    name='Calibración del modelo',
    hovertemplate="Pred: %{x:.2%}<br>Real: %{y:.2%}<extra></extra>",
))
fig.update_layout(
    xaxis_title="Probabilidad predicha (media por bin)",
    yaxis_title="Tasa de devolución real (media por bin)",
    xaxis=dict(tickformat='.0%'),
    yaxis=dict(tickformat='.0%'),
)
st.plotly_chart(plotly_layout(fig, height=420), use_container_width=True)

# ============================================================
# TOP 50 PRODUCTOS DE MAYOR RIESGO
# ============================================================
section_divider()
st.markdown("## Top 50 productos de mayor riesgo")
caption(
    "Los 50 artículos con mayor probabilidad de devolución predicha. Esta "
    "tabla es accionable: candidatos prioritarios para reducir producción "
    "o rediseñar (tallaje, descripción, fotos)."
)

top50 = df.nlargest(50, 'avg_return_prob')[[
    'article_id', 'product_type_name', 'colour_group_name',
    'garment_group_name', 'department_name',
    'avg_return_prob', 'actual_return_rate', 'n_transactions',
]].copy()

top50.columns = [
    'Article ID', 'Product type', 'Color', 'Garment', 'Departamento',
    'Predicted', 'Actual', 'N transactions',
]
top50['Predicted'] = (top50['Predicted'] * 100).round(1).astype(str) + '%'
top50['Actual'] = (top50['Actual'] * 100).round(1).astype(str) + '%'

st.dataframe(top50, use_container_width=True, hide_index=True, height=420)

# Botón de descarga (acción de negocio real)
csv = top50.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇  Descargar lista (CSV)",
    data=csv,
    file_name="top50_productos_riesgo_devolucion.csv",
    mime="text/csv",
)

# ============================================================
# Footer
# ============================================================
st.markdown(
    f"""
    <div style='text-align:center; color:{COLORS["muted"]};
                font-size:0.8rem; margin-top:3rem; padding-top:1.5rem;
                border-top: 1px solid {COLORS["border"]};'>
        Returns Risk · {len(df):,} articles · ROC-AUC {model_metrics['roc_auc']:.3f}
    </div>
    """,
    unsafe_allow_html=True,
)
