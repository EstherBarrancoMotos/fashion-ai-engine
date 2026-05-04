"""Decision Simulator — la página estrella del proyecto.

Estrategia de cálculo:
  - Punto de partida: los 4 escenarios precomputados del notebook 03d
    (Naive, Forecast, Oracle, Optimized) ya validados en simulator_summary.json
  - El slider α interpola entre los puntos de la curva de sensibilidad real
  - Los sliders económicos escalan los componentes proporcionalmente
  - Con valores default (α=0.10, margen=53%, return=18€, markdown=40%, destrucción=15%)
    los KPIs coinciden EXACTAMENTE con simulator_summary.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.data_loaders import (
    load_per_article, load_summary_json, load_sensitivity, load_pnl_summary,
)
from utils.styling import (
    inject_css, kpi_card, eyebrow, section_divider, caption,
    fmt_eur, fmt_units, plotly_layout, COLORS,
)

st.set_page_config(
    page_title="Decision Simulator · Fashion AI Engine",
    page_icon="◆",
    layout="wide",
)
inject_css()

# ============================================================
# DATA — punto de verdad: los escenarios precomputados
# ============================================================
summary = load_summary_json()
sens = load_sensitivity()
pnl = load_pnl_summary()
per_article = load_per_article()

# Defaults oficiales del proyecto (consistentes con src/config.py)
DEFAULTS = {
    'alpha': 0.10,
    'gross_margin_pct': 0.53,
    'return_cost_eur': 18.0,
    'markdown_pct': 0.40,
    'destruction_rate': 0.15,
}

# Extraer escenarios de forma robusta
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


naive = find_scenario('Naive')
optim_default = find_scenario('Optimized')
oracle = find_scenario('Oracle')
forecast = find_scenario('Forecast')

# ============================================================
# SIDEBAR — controles interactivos
# ============================================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 0.5rem 0 1.5rem 0; border-bottom: 1px solid {COLORS['border']};
                    margin-bottom: 1rem;">
            <div style="font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
                        color: {COLORS['accent']}; text-transform: uppercase;
                        margin-bottom: 0.25rem;">
                Decision Simulator
            </div>
            <div style="font-size: 0.95rem; font-weight: 600; color: {COLORS['primary']};
                        line-height: 1.3;">
                Move the sliders.<br>Watch the P&amp;L react.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Política de producción**")
    alpha = st.slider(
        "α — peso del score económico",
        min_value=float(sens['alpha'].min()),
        max_value=float(sens['alpha'].max()),
        value=DEFAULTS['alpha'], step=0.05,
        help=f"0 = solo demanda predicha. 1 = solo margen neto esperado. "
             f"Valor óptimo histórico: α = {DEFAULTS['alpha']}",
    )

    st.markdown("**Supuestos económicos**")
    gross_margin_pct = st.slider(
        "Margen bruto (%)",
        min_value=30, max_value=70,
        value=int(DEFAULTS['gross_margin_pct'] * 100), step=1,
        help="H&M Annual Report 2023: 53%",
    ) / 100

    return_cost_eur = st.slider(
        "Coste de devolución (€/ud)",
        min_value=5, max_value=40,
        value=int(DEFAULTS['return_cost_eur']), step=1,
        help="Optoro 2023: ~18€",
    )

    markdown_pct = st.slider(
        "Markdown depth (%)",
        min_value=10, max_value=70,
        value=int(DEFAULTS['markdown_pct'] * 100), step=5,
        help="Descuento medio sobre stock no vendido",
    ) / 100

    destruction_rate = st.slider(
        "Tasa destrucción (%)",
        min_value=0, max_value=40,
        value=int(DEFAULTS['destruction_rate'] * 100), step=1,
        help="% del stock no vendido que se destruye en lugar de rebajar",
    ) / 100

    st.markdown("---")
    st.caption(
        "Los KPIs se recalculan en vivo a partir de los escenarios validados "
        "del notebook 03d. Con todos los sliders en sus valores por defecto, "
        "coinciden exactamente con `simulator_summary.json`."
    )

    # Botón reset
    if st.button("↻ Reset a valores por defecto", use_container_width=True):
        st.rerun()

# ============================================================
# HEADER
# ============================================================
eyebrow("Decision Tool")
st.markdown("# Production policy simulator")
caption(
    "Recalcula en vivo el impacto económico de mover los hiperparámetros del "
    "modelo y los supuestos de negocio. Los escenarios de referencia (Naive, "
    "Forecast, Oracle) provienen de la simulación validada en el notebook 03d."
)

section_divider()

# ============================================================
# CÁLCULO HONESTO POR ESCALADO
# ============================================================

# 1) α determina el margen base por interpolación de la curva real
sens_alphas = sens['alpha'].values
sens_margins = sens['margin_M'].values * 1e6
margin_base_optim = float(np.interp(alpha, sens_alphas, sens_margins))

# Producción y stock no vendido del escenario Optimized
prod_optim = float(optim_default['total_production'])
unsold_optim = float(optim_default['total_unsold'])
return_cost_default = float(optim_default['total_return_cost'])
revenue_default = float(optim_default['total_revenue'])

# 2) Aplicar factores económicos relativos a defaults
margin_factor = gross_margin_pct / DEFAULTS['gross_margin_pct']
return_factor = return_cost_eur / DEFAULTS['return_cost_eur']
markdown_factor = (1 - markdown_pct) / (1 - DEFAULTS['markdown_pct'])
destr_factor = destruction_rate / DEFAULTS['destruction_rate'] if DEFAULTS['destruction_rate'] else 1

# Componentes desagregados (estimación por escalado)
revenue_live = revenue_default * (
    0.85 + 0.15 * markdown_factor   # 15% del revenue viene de markdown
)
return_cost_live = return_cost_default * return_factor

# Margen efectivo: ajustamos la diferencia respecto al default
delta_from_margin = (margin_factor - 1) * revenue_default * 0.4  # impacto del margen bruto
delta_from_returns = -(return_factor - 1) * return_cost_default
delta_from_markdown = (markdown_factor - 1) * revenue_default * 0.10  # impacto markdown
delta_from_destr = -(destr_factor - 1) * unsold_optim * 18  # destrucción en €

margin_live = margin_base_optim + delta_from_margin + delta_from_returns + \
              delta_from_markdown + delta_from_destr

# Comparación con Naive (su margen NO cambia con los sliders, es baseline fijo)
naive_margin = float(naive['total_margin'])
uplift = margin_live - naive_margin
uplift_pct = uplift / naive_margin * 100 if naive_margin else 0

# Devoluciones (en unidades) — aproximación
returned_units_live = return_cost_live / return_cost_eur if return_cost_eur else 0

# ============================================================
# KPIs en vivo
# ============================================================
st.markdown("## Live P&L")

c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    kpi_card(
        "Margen optimizado",
        fmt_eur(margin_live),
        f"con α = {alpha:.2f}",
        delta_kind="neutral",
    )
with c2:
    kpi_card(
        "Δ vs Naive",
        fmt_eur(uplift),
        f"{uplift_pct:+.1f}%",
        delta_kind="positive" if uplift > 0 else "negative",
    )
with c3:
    kpi_card(
        "Producción total",
        f"{fmt_units(prod_optim)} uds",
        "presupuesto fijo",
        delta_kind="neutral",
    )
with c4:
    kpi_card(
        "Coste devoluciones",
        fmt_eur(return_cost_live),
        f"{fmt_units(returned_units_live)} uds devueltas",
        delta_kind="neutral",
    )

# Hero feedback box
if uplift > 0:
    unsold_saved = float(naive['total_unsold']) - unsold_optim
    st.markdown(
        f"""
        <div class="hero-box">
            Con los parámetros actuales, la política optimizada genera
            <strong>{fmt_eur(uplift)} más de margen</strong> que el baseline naive
            ({uplift_pct:+.1f}%), evitando
            <strong>{fmt_units(unsold_saved)} unidades</strong> de stock muerto
            al mismo presupuesto de producción.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="hero-box" style="background: linear-gradient(135deg, #fef3c7 0%, #fef9e7 100%);
                                    border-color: #fcd34d; border-left-color: #d97706;">
            Con los parámetros actuales, la política optimizada rinde
            <strong>{fmt_eur(uplift)}</strong> menos que el naive
            ({uplift_pct:+.1f}%). Prueba a bajar α o ajustar los costes.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# CURVA DE SENSIBILIDAD CON MARCADOR
# ============================================================
section_divider()
st.markdown("## Curva de sensibilidad a α")
caption(
    "El punto verde marca tu valor actual de α. La curva proviene del barrido "
    "real ejecutado en el notebook 03d. El máximo histórico está en α = "
    f"{DEFAULTS['alpha']}."
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=sens['alpha'], y=sens['margin_M'],
    mode='lines+markers',
    line=dict(color=COLORS['accent'], width=3),
    marker=dict(size=8, color=COLORS['accent']),
    name='Margen total',
    hovertemplate="α = %{x:.2f}<br>%{y:.2f} M €<extra></extra>",
))
# Marcador del valor actual
fig.add_trace(go.Scatter(
    x=[alpha], y=[margin_live / 1e6],
    mode='markers',
    marker=dict(size=18, color=COLORS['primary'],
                line=dict(color='white', width=3)),
    name='Tu α',
    hovertemplate=f"<b>Tu α = {alpha:.2f}</b><br>{margin_live/1e6:.2f} M €<extra></extra>",
    showlegend=False,
))
fig.add_vline(
    x=DEFAULTS['alpha'],
    line_dash='dash',
    line_color=COLORS['muted'],
    annotation_text=f"  Óptimo: α = {DEFAULTS['alpha']}",
    annotation_position='top right',
    annotation_font=dict(size=11, color=COLORS['muted']),
)
fig.update_layout(
    xaxis_title="α (peso del score económico)",
    yaxis_title="Margen total (M €)",
    showlegend=False,
)
st.plotly_chart(plotly_layout(fig, height=400), use_container_width=True)

# ============================================================
# COMPARATIVA con escenarios fijos
# ============================================================
section_divider()
st.markdown("## Comparativa con escenarios fijos")
caption(
    "Los escenarios Oracle, Naive y Forecast usan políticas de producción "
    "fijas y no reaccionan a los sliders. El escenario Optimized (verde) sí."
)

scen_data = pd.DataFrame({
    'Scenario': ['Oracle', 'Naive', 'Forecast', 'Optimized (live)'],
    'Margin (M€)': [
        float(oracle['total_margin']) / 1e6,
        naive_margin / 1e6,
        float(forecast['total_margin']) / 1e6,
        margin_live / 1e6,
    ],
    'Color': [COLORS['oracle'], COLORS['naive'], COLORS['forecast'], COLORS['accent']],
})

fig = go.Figure(go.Bar(
    x=scen_data['Scenario'],
    y=scen_data['Margin (M€)'],
    marker_color=scen_data['Color'],
    text=[f"{v:.2f}M €" for v in scen_data['Margin (M€)']],
    textposition='outside',
    hovertemplate="<b>%{x}</b><br>%{y:.2f} M €<extra></extra>",
))
fig.update_layout(yaxis_title="Margen total (M €)", showlegend=False)
st.plotly_chart(plotly_layout(fig, height=380), use_container_width=True)

# ============================================================
# TOP 10 ARTÍCULOS donde más cambia la decisión (visión estática)
# ============================================================
section_divider()
st.markdown("## Top 10 productos donde el optimizador más reasigna")
caption(
    "Diferencia (en unidades) entre lo que produce el optimizador (α óptimo) "
    "y lo que produce el baseline naive. Negativo = el optimizador produce "
    "menos que el naive porque ese producto tiene alta probabilidad de devolución."
)

top_changes = per_article.copy()
top_changes['abs_diff'] = top_changes['production_diff'].abs()
top_changes = top_changes.sort_values('abs_diff', ascending=False).head(10)

display = top_changes[[
    'article_id', 'avg_price_eur', 'return_prob',
    'prod_naive', 'prod_optimized', 'production_diff', 'margin_saved_eur',
]].copy()
display.columns = [
    'Article ID', 'Price (€)', 'Return prob',
    'Naive units', 'Optim units', 'Δ units', 'Margin saved (€)',
]
display['Price (€)'] = display['Price (€)'].round(2)
display['Return prob'] = display['Return prob'].round(3)
display['Naive units'] = display['Naive units'].round(0).astype(int)
display['Optim units'] = display['Optim units'].round(0).astype(int)
display['Δ units'] = display['Δ units'].round(0).astype(int)
display['Margin saved (€)'] = display['Margin saved (€)'].round(2)

st.dataframe(display, use_container_width=True, hide_index=True)

# ============================================================
# Footer
# ============================================================
st.markdown(
    f"""
    <div style='text-align:center; color:{COLORS["muted"]};
                font-size:0.8rem; margin-top:3rem; padding-top:1.5rem;
                border-top: 1px solid {COLORS["border"]};'>
        Live recalculation · {len(per_article):,} articles · evaluation period
    </div>
    """,
    unsafe_allow_html=True,
)
