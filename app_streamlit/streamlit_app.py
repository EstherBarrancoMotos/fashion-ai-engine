"""Fashion AI Engine — Home page (Executive Dashboard).

Vista ejecutiva del motor de decisión:
- KPIs principales del impacto vs. baseline humano
- Comparativa visual de los 4 escenarios (producción, stock, margen)
- Sensibilidad al hiperparámetro alpha
- Caveats y metodología
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import plotly.graph_objects as go

from utils.data_loaders import load_pnl_summary, load_summary_json, load_sensitivity
from utils.styling import (
    inject_css, kpi_card, eyebrow, section_divider, caption,
    fmt_eur, fmt_units, plotly_layout, COLORS,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Fashion AI Engine",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
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
                Fashion AI Engine
            </div>
            <div style="font-size: 1.1rem; font-weight: 700; color: {COLORS['primary']};
                        line-height: 1.2;">
                Decision intelligence<br>for fashion brands
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HEADER
# ============================================================
eyebrow("Executive Dashboard")
st.markdown("# From overproduction to optimization")
caption(
    "A modular ML decision engine that combines demand forecasting, "
    "return-risk modeling and economic simulation to quantify, in euros, "
    "the impact of smarter production policies."
)

section_divider()

# ============================================================
# DATA LOAD
# ============================================================
summary = load_summary_json()
pnl = load_pnl_summary()
sens = load_sensitivity()

# Extract scenarios robustly
raw_scenarios = summary.get('scenarios')
if isinstance(raw_scenarios, list):
    scenarios = {s['scenario']: s for s in raw_scenarios}
elif isinstance(raw_scenarios, dict):
    scenarios = raw_scenarios
else:
    scenarios = {row['scenario']: row.to_dict() for _, row in pnl.iterrows()}


def find_scenario(prefix: str) -> dict:
    for k, v in scenarios.items():
        if k.startswith(prefix):
            return v
    raise KeyError(f"Scenario '{prefix}' not found")


opt = find_scenario('Optimized')
naive = find_scenario('Naive')
forecast = find_scenario('Forecast')
oracle = find_scenario('Oracle')

# Read or compute impact (compatible with both old and new JSON shapes)
impact = summary.get('impact_vs_naive') or summary.get('impact')
if not impact:
    margin_uplift_eur = float(opt['total_margin'] - naive['total_margin'])
    impact = {
        'margin_uplift_eur': margin_uplift_eur,
        'margin_uplift_pct': margin_uplift_eur / float(naive['total_margin']) * 100,
        'unsold_saved_units': float(naive['total_unsold'] - opt['total_unsold']),
    }

alpha_optimal = float(summary.get('alpha_optimal', summary.get('best_alpha', 0.1)))
gap_to_oracle = float(
    summary.get('gap_to_oracle_eur',
                oracle['total_margin'] - opt['total_margin'])
)

# ============================================================
# KPIs
# ============================================================
st.markdown("## Impact at a glance")

c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    kpi_card(
        "Margen incremental",
        fmt_eur(impact['margin_uplift_eur']),
        f"{impact.get('margin_uplift_pct', 0):+.1f}% vs forecast humano",
        delta_kind="positive",
    )
with c2:
    unsold = impact.get('unsold_saved_units', 0)
    kpi_card(
        "Stock muerto evitado",
        f"{fmt_units(unsold)} uds",
        "vs baseline naive",
        delta_kind="positive" if unsold > 0 else "neutral",
    )
with c3:
    kpi_card(
        "Alpha óptimo",
        f"α = {alpha_optimal:.1f}",
        "peso del score económico",
        delta_kind="neutral",
    )
with c4:
    kpi_card(
        "Gap vs Oracle",
        fmt_eur(gap_to_oracle),
        "techo teórico de mejora",
        delta_kind="neutral",
    )

# Hero summary
st.markdown(
    f"""
    <div class="hero-box">
        <strong>Resultado clave.</strong> Reasignar la producción entre artículos
        según su margen neto esperado — no según volumen — genera
        <strong>{fmt_eur(impact['margin_uplift_eur'])}</strong> de margen
        adicional <em>al mismo presupuesto de unidades</em> que el baseline humano.
        El optimizador no produce más: <strong>produce mejor</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SCENARIO COMPARISON
# ============================================================
section_divider()
st.markdown("## The four scenarios")
caption(
    "Mismo dataset de artículos, mismo periodo de evaluación. Cuatro políticas "
    "de producción distintas, evaluadas con un modelo económico común."
)

# Order scenarios consistently
order_keys = ['Oracle', 'Naive', 'Forecast', 'Optimized']
ordered_names = []
for prefix in order_keys:
    for name in pnl['scenario']:
        if name.startswith(prefix):
            ordered_names.append(name)
            break

color_seq = [COLORS['oracle'], COLORS['naive'], COLORS['forecast'], COLORS['optimized']]
pnl_ord = pnl.set_index('scenario').loc[ordered_names].reset_index()

col_a, col_b, col_c = st.columns(3, gap="medium")

with col_a:
    fig = go.Figure(go.Bar(
        x=pnl_ord['scenario'],
        y=pnl_ord['total_production'] / 1e6,
        marker_color=color_seq,
        text=[f"{v/1e6:.2f}M" for v in pnl_ord['total_production']],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>%{y:.2f}M unidades<extra></extra>",
    ))
    fig.update_layout(
        title="Producción total",
        yaxis_title="M unidades",
        showlegend=False,
    )
    fig.update_xaxes(tickangle=20)
    st.plotly_chart(plotly_layout(fig), use_container_width=True)

with col_b:
    fig = go.Figure(go.Bar(
        x=pnl_ord['scenario'],
        y=pnl_ord['total_unsold'] / 1e3,
        marker_color=color_seq,
        text=[f"{v/1e3:.0f}K" for v in pnl_ord['total_unsold']],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>%{y:.0f}K unidades<extra></extra>",
    ))
    fig.update_layout(
        title="Stock no vendido",
        yaxis_title="K unidades",
        showlegend=False,
    )
    fig.update_xaxes(tickangle=20)
    st.plotly_chart(plotly_layout(fig), use_container_width=True)

with col_c:
    fig = go.Figure(go.Bar(
        x=pnl_ord['scenario'],
        y=pnl_ord['total_margin'] / 1e6,
        marker_color=color_seq,
        text=[f"{v/1e6:.2f}M €" for v in pnl_ord['total_margin']],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>%{y:.2f}M €<extra></extra>",
    ))
    fig.update_layout(
        title="Margen total",
        yaxis_title="M €",
        showlegend=False,
    )
    fig.update_xaxes(tickangle=20)
    st.plotly_chart(plotly_layout(fig), use_container_width=True)

# ============================================================
# ALPHA SENSITIVITY
# ============================================================
section_divider()
st.markdown("## Sensibilidad al parámetro α")
caption(
    "α controla cuánto pesa el score económico (margen neto esperado por unidad) "
    "frente al volumen de demanda predicho. Penalizaciones más agresivas destruyen "
    "valor al desviar producción de productos populares."
)

best_alpha_row = sens.loc[sens['margin_M'].idxmax()]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=sens['alpha'], y=sens['margin_M'],
    mode='lines+markers',
    line=dict(color=COLORS['accent'], width=3),
    marker=dict(size=9, color=COLORS['accent']),
    name='Margen total',
    hovertemplate="α = %{x:.1f}<br>%{y:.2f} M €<extra></extra>",
))
fig.add_vline(
    x=best_alpha_row['alpha'],
    line_dash='dash',
    line_color=COLORS['accent_dark'],
    annotation_text=f"  Óptimo: α = {best_alpha_row['alpha']:.1f}",
    annotation_position='top right',
    annotation_font=dict(size=12, color=COLORS['accent_dark']),
)
fig.update_layout(
    title=None,
    xaxis_title="α (peso del score económico)",
    yaxis_title="Margen total (M €)",
    showlegend=False,
)
st.plotly_chart(plotly_layout(fig, height=400), use_container_width=True)

# ============================================================
# METHODOLOGY
# ============================================================
section_divider()
with st.expander("Metodología y limitaciones honestas", expanded=False):
    st.markdown(f"""
    **Dataset.** H&M Personalized Fashion Recommendations (Kaggle competition, 2022).
    31.8M transacciones, 105K artículos, septiembre 2018 – septiembre 2020.

    **Modelos.**
    - **Demanda**: LightGBM con log-target, validación temporal estricta,
      categoricals nativos. WAPE = 0.515 sobre test (mejora 7.5% sobre baseline `lag_1`).
    - **Devoluciones**: LightGBM classifier sobre labels semi-sintéticas
      (tasas sectoriales Narvar / Optoro). ROC-AUC = 0.64.
    - **Clustering**: KMeans k=4 sobre atributos categóricos one-hot,
      seleccionado por silhouette.
    - **Estudio comparativo**: 5 algoritmos supervisados (Dummy, Ridge, RandomForest,
      XGBoost, LightGBM) en `Pipeline + GridSearchCV` con `TimeSeriesSplit(3)`.

    **Supuestos económicos** (`src/config.py`):
    - Margen bruto: 53% (H&M Annual Report 2023)
    - Coste devolución: 18 €/ud (Optoro 2023)
    - Markdown medio: 40% (McKinsey State of Fashion 2024)
    - Tasa destrucción stock muerto: 15%

    **Escenarios.**
    - **Oracle (perfect info)**: produce `actual_demand × 1.2`. Techo teórico —
      no comparable de forma realista.
    - **Naive baseline**: forecast humano = demanda real distorsionada con error
      gaussiano (σ=25%). Simula la decisión de un buyer sin ML.
    - **Forecast-only (ML)**: predicción del modelo de demanda directa, sin
      lógica de devoluciones.
    - **Optimized (smart, α={alpha_optimal})**: reasigna el mismo presupuesto de
      unidades del Oracle ponderando por margen neto esperado por artículo.

    **Limitaciones honestas.**
    - Las labels de devolución son semi-sintéticas (H&M no publica devoluciones).
      El modelo aprende patrones realistas pero está sesgado a las tasas asumidas.
    - El simulador es estático: sin sustitución entre artículos, sin dinámicas
      inter-temporales de inventario, sin elasticidad de precio.
    - El cap de demanda (2.5× predicción) es un parámetro, no derivado de datos.
    """)

# Footer
st.markdown(
    f"""
    <div style='text-align:center; color:{COLORS["muted"]};
                font-size:0.8rem; margin-top:3rem; padding-top:1.5rem;
                border-top: 1px solid {COLORS["border"]};'>
        Fashion AI Engine · Built by Esther Barranco Motos · Portfolio project
    </div>
    """,
    unsafe_allow_html=True,
)
