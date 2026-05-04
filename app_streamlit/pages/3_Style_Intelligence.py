"""Style Intelligence — análisis cruzado de prenda, color, cluster y devoluciones.

Cruza:
  - return_predictions.parquet (avg_return_prob por artículo)
  - article_features.parquet (atributos categóricos)
  - article_clusters.parquet (cluster KMeans)
  - simulator_per_article.parquet (margen reasignado, opcional)

Responde: ¿qué tipo de prenda devuelve más? ¿qué color? ¿qué cluster
concentra el problema? ¿dónde están las oportunidades de margen?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils.data_loaders import (
    load_return_predictions, load_article_features, load_per_article,
)
from utils.styling import (
    inject_css, kpi_card, eyebrow, section_divider, caption,
    fmt_eur, fmt_units, plotly_layout, COLORS,
)

st.set_page_config(
    page_title="Style Intelligence · Fashion AI Engine",
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
                Style Intelligence
            </div>
            <div style="font-size: 0.95rem; font-weight: 600; color: {COLORS['primary']};
                        line-height: 1.3;">
                Where the risk<br>and the margin live.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# DATA LOADING + JOIN
# ============================================================
returns = load_return_predictions().copy()
features = load_article_features().copy()

# article_clusters tiene su propio loader minimalista (sin caché compartida)
@st.cache_data
def load_clusters():
    p = Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'article_clusters.parquet'
    df = pd.read_parquet(p)
    df['article_id'] = df['article_id'].astype(str).str.zfill(10)
    return df


clusters = load_clusters()
returns['article_id'] = returns['article_id'].astype(str).str.zfill(10)

# Merge maestro
df = (returns
      .merge(features, on='article_id', how='left')
      .merge(clusters, on='article_id', how='left'))

# Estadísticas globales
total_articles = len(df)
mean_return = df['avg_return_prob'].mean()
n_clusters = int(df['cluster'].nunique())

# ============================================================
# HEADER
# ============================================================
eyebrow("Style Insights")
st.markdown("# Where the risk and the margin live")
caption(
    "Cruce de la probabilidad de devolución predicha con los atributos de "
    "producto (tipo de prenda, color, departamento) y los clusters de estilo "
    "identificados por KMeans. El objetivo: identificar dónde está concentrado "
    "el riesgo y dónde están las oportunidades."
)

section_divider()

# ============================================================
# KPIs GLOBALES
# ============================================================
st.markdown("## Vista general")

c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    kpi_card(
        "Productos analizados",
        fmt_units(total_articles),
        f"con predicción de devolución",
        delta_kind="neutral",
    )
with c2:
    kpi_card(
        "Tasa devolución media",
        f"{mean_return*100:.1f}%",
        "media ponderada por artículo",
        delta_kind="neutral",
    )
with c3:
    kpi_card(
        "Clusters de estilo",
        f"{n_clusters}",
        "vía KMeans (silhouette)",
        delta_kind="neutral",
    )
with c4:
    high_risk = (df['avg_return_prob'] > 0.30).sum()
    pct_high = high_risk / total_articles * 100
    kpi_card(
        "Productos de alto riesgo",
        fmt_units(high_risk),
        f"{pct_high:.1f}% con prob > 30%",
        delta_kind="negative",
    )

# ============================================================
# HEATMAP — GARMENT GROUP × COLOR
# ============================================================
section_divider()
st.markdown("## Mapa de riesgo: tipo de prenda × color")
caption(
    "Tasa media de devolución cruzando garment group × colour. Verde = bajo "
    "riesgo, rojo = alto riesgo. Solo se muestran combinaciones con al menos "
    "20 productos para evitar ruido."
)

# Top N de cada para que el heatmap sea legible
top_garments = (df.groupby('garment_group_name', observed=True).size()
                  .sort_values(ascending=False).head(12).index.tolist())
top_colors = (df.groupby('colour_group_name', observed=True).size()
                .sort_values(ascending=False).head(15).index.tolist())

sub = df[df['garment_group_name'].isin(top_garments) &
         df['colour_group_name'].isin(top_colors)].copy()

heat = sub.groupby(['garment_group_name', 'colour_group_name'], observed=True).agg(
    return_rate=('avg_return_prob', 'mean'),
    n=('article_id', 'count'),
).reset_index()

# Filtrar combinaciones con n >= 20
heat = heat[heat['n'] >= 20]

# Pivot para heatmap
heat_pivot = heat.pivot(index='garment_group_name',
                        columns='colour_group_name',
                        values='return_rate')

fig = go.Figure(go.Heatmap(
    z=heat_pivot.values * 100,
    x=heat_pivot.columns,
    y=heat_pivot.index,
    colorscale=[
        [0.0, COLORS['accent']],
        [0.5, '#f59e0b'],
        [1.0, COLORS['bad']],
    ],
    colorbar=dict(title="Return %", thickness=12),
    hovertemplate="<b>%{y}</b> en <b>%{x}</b><br>Return rate: %{z:.1f}%<extra></extra>",
    text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in heat_pivot.values * 100],
    texttemplate="%{text}",
    textfont={"size": 9, "color": "white"},
))
fig.update_layout(
    xaxis_title=None,
    yaxis_title=None,
    xaxis_tickangle=-30,
)
st.plotly_chart(plotly_layout(fig, height=500), use_container_width=True)

# ============================================================
# TOP 10 GARMENT GROUPS
# ============================================================
section_divider()
st.markdown("## Top categorías por tasa de devolución")
caption(
    "Garment groups ordenados por tasa media de devolución predicha. "
    "Las que están en rojo concentran el riesgo: candidatos a reducir producción."
)

garment_stats = df.groupby('garment_group_name', observed=True).agg(
    n_products=('article_id', 'count'),
    return_rate=('avg_return_prob', 'mean'),
).reset_index()
garment_stats = garment_stats[garment_stats['n_products'] >= 30]
garment_stats = garment_stats.sort_values('return_rate', ascending=True).tail(15)

# Color por nivel de riesgo
def risk_color(rate):
    if rate < 0.15: return COLORS['accent']
    if rate < 0.25: return '#f59e0b'
    return COLORS['bad']

bar_colors = [risk_color(r) for r in garment_stats['return_rate']]

fig = go.Figure(go.Bar(
    x=garment_stats['return_rate'] * 100,
    y=garment_stats['garment_group_name'],
    orientation='h',
    marker_color=bar_colors,
    text=[f"{r*100:.1f}%  ·  {int(n):,} productos" for r, n in
          zip(garment_stats['return_rate'], garment_stats['n_products'])],
    textposition='outside',
    hovertemplate="<b>%{y}</b><br>Return rate: %{x:.1f}%<extra></extra>",
))
fig.update_layout(
    xaxis_title="Tasa de devolución predicha (%)",
    yaxis_title=None,
    showlegend=False,
)
st.plotly_chart(plotly_layout(fig, height=520), use_container_width=True)

# ============================================================
# ANÁLISIS POR CLUSTER
# ============================================================
section_divider()
st.markdown("## Perfil de los clusters de estilo")
caption(
    "Cada cluster agrupa productos con perfil similar de atributos categóricos "
    "(KMeans k=4). Vemos cuántos productos contiene cada uno, qué garment "
    "domina y cuál es su tasa de devolución."
)

cluster_profile = df.groupby('cluster', observed=True).agg(
    n_products=('article_id', 'count'),
    return_rate=('avg_return_prob', 'mean'),
    top_garment=('garment_group_name', lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A'),
    top_color=('colour_group_name', lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A'),
    top_dept=('department_name', lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A'),
).reset_index()

cluster_profile['cluster_label'] = 'Cluster ' + cluster_profile['cluster'].astype(int).astype(str)

# Display nice
display_clusters = cluster_profile[[
    'cluster_label', 'n_products', 'return_rate',
    'top_garment', 'top_color', 'top_dept',
]].copy()
display_clusters.columns = [
    'Cluster', 'Productos', 'Return rate', 'Top garment', 'Top color', 'Top departamento',
]
display_clusters['Return rate'] = (display_clusters['Return rate'] * 100).round(1).astype(str) + '%'
display_clusters['Productos'] = display_clusters['Productos'].apply(lambda x: f"{x:,}")

st.dataframe(display_clusters, use_container_width=True, hide_index=True)

# ============================================================
# DISTRIBUCIÓN POR CLUSTER (chart)
# ============================================================
col_a, col_b = st.columns(2, gap="medium")

with col_a:
    fig = go.Figure(go.Bar(
        x=cluster_profile['cluster_label'],
        y=cluster_profile['n_products'],
        marker_color=[COLORS['accent'], COLORS['forecast'], COLORS['oracle'], COLORS['naive']][:n_clusters],
        text=[f"{int(n):,}" for n in cluster_profile['n_products']],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>%{y:,} productos<extra></extra>",
    ))
    fig.update_layout(
        title="Volumen de productos por cluster",
        yaxis_title="Productos",
        showlegend=False,
    )
    st.plotly_chart(plotly_layout(fig, height=380), use_container_width=True)

with col_b:
    fig = go.Figure(go.Bar(
        x=cluster_profile['cluster_label'],
        y=cluster_profile['return_rate'] * 100,
        marker_color=[risk_color(r) for r in cluster_profile['return_rate']],
        text=[f"{r*100:.1f}%" for r in cluster_profile['return_rate']],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Return rate: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Tasa de devolución por cluster",
        yaxis_title="Return rate (%)",
        showlegend=False,
    )
    st.plotly_chart(plotly_layout(fig, height=380), use_container_width=True)

# ============================================================
# EXPLORACIÓN INTERACTIVA — filtro por garment group
# ============================================================
section_divider()
st.markdown("## Explorar una categoría")
caption(
    "Selecciona un garment group para ver su distribución de probabilidad "
    "de devolución y los 10 productos con mayor riesgo dentro de él."
)

selected_garment = st.selectbox(
    "Garment group",
    options=sorted(df['garment_group_name'].dropna().unique().tolist()),
    index=0,
)

sub = df[df['garment_group_name'] == selected_garment].copy()

col1, col2 = st.columns([1, 1.3], gap="medium")

with col1:
    # Histograma de return_prob
    fig = go.Figure(go.Histogram(
        x=sub['avg_return_prob'] * 100,
        nbinsx=25,
        marker_color=COLORS['accent'],
        marker_line_color='white',
        marker_line_width=1,
    ))
    fig.update_layout(
        title=f"Distribución de return probability · {selected_garment}",
        xaxis_title="Return probability (%)",
        yaxis_title="N productos",
        showlegend=False,
    )
    st.plotly_chart(plotly_layout(fig, height=380), use_container_width=True)

with col2:
    # Top 10 productos de mayor riesgo en esa categoría
    top_risk = sub.nlargest(10, 'avg_return_prob')[[
        'article_id', 'colour_group_name', 'department_name', 'avg_return_prob',
    ]].copy()
    top_risk.columns = ['Article ID', 'Color', 'Departamento', 'Return prob']
    top_risk['Return prob'] = (top_risk['Return prob'] * 100).round(1).astype(str) + '%'
    st.markdown(f"**Top 10 productos de mayor riesgo en {selected_garment}**")
    st.dataframe(top_risk, use_container_width=True, hide_index=True, height=380)

# ============================================================
# Footer
# ============================================================
st.markdown(
    f"""
    <div style='text-align:center; color:{COLORS["muted"]};
                font-size:0.8rem; margin-top:3rem; padding-top:1.5rem;
                border-top: 1px solid {COLORS["border"]};'>
        Style Intelligence · {total_articles:,} articles · {n_clusters} KMeans clusters
    </div>
    """,
    unsafe_allow_html=True,
)
