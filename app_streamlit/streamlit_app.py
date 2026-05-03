"""
Fashion AI Engine — Executive Dashboard
Home page del dashboard de decisión para reducir sobreproducción.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import plotly.graph_objects as go

from utils.data_loaders import load_pnl_summary, load_summary_json, load_sensitivity
from utils.styling import inject_css, kpi_card, fmt_eur, fmt_units, COLORS

st.set_page_config(
    page_title="Fashion AI Engine",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ============================================================
# HEADER
# ============================================================
st.markdown("# 👗 Fashion AI Engine")
st.markdown(
    "**Motor de decisión para reducir sobreproducción en marcas de moda.**  \n"
    "Combina forecast de demanda + predicción de devoluciones + simulador económico "
    "para optimizar la distribución de producción a presupuesto constante."
)
st.markdown("---")

# ============================================================
# DATA — robust to both JSON shapes (list-of-dicts or dict-of-dicts)
# ============================================================
summary = load_summary_json()
pnl = load_pnl_summary()
sens = load_sensitivity()

# Extract scenarios into a uniform dict {name: row_dict}
raw_scenarios = summary.get('scenarios')
if isinstance(raw_scenarios, list):
    scenarios = {s['scenario']: s for s in raw_scenarios}
elif isinstance(raw_scenarios, dict):
    scenarios = raw_scenarios
else:
    # Fallback: build from pnl parquet
    scenarios = {row['scenario']: row.to_dict() for _, row in pnl.iterrows()}

def find_scenario(prefix: str) -> dict:
    for k, v in scenarios.items():
        if k.startswith(prefix):
            return v
    raise KeyError(f"Scenario starting with '{prefix}' not found")

opt = find_scenario('Optimized')
naive = find_scenario('Naive')
forecast = find_scenario('Forecast')
oracle = find_scenario('Oracle')

# Read or compute impact
impact = summary.get('impact_vs_naive')
if not impact:
    margin_uplift_eur = float(opt.get('total_margin', 0) - naive.get('total_margin', 0))
    margin_uplift_pct = margin_uplift_eur / float(naive.get('total_margin', 1)) * 100
    returns_saved_eur = float(naive.get('total_return_cost', 0) - opt.get('total_return_cost', 0))
    unsold_saved_units = float(naive.get('total_unsold', 0) - opt.get('total_unsold', 0))
    impact = {
        'margin_uplift_eur': margin_uplift_eur,
        'margin_uplift_pct': margin_uplift_pct,
        'returns_saved_eur': returns_saved_eur,
        'unsold_saved_units': unsold_saved_units,
    }

alpha_optimal = float(summary.get('alpha_optimal', 0.1))
gap_to_oracle = float(summary.get('gap_to_oracle_eur',
                                  oracle.get('total_margin', 0) - opt.get('total_margin', 0)))

# ============================================================
# KPIs
# ============================================================
st.markdown("### 📊 Impacto del motor optimizado")

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card(
        "Margen adicional",
        fmt_eur(impact['margin_uplift_eur']),
        f"{impact['margin_uplift_pct']:+.1f}% vs forecast humano",
        delta_positive=True,
    )
with c2:
    kpi_card(
        "Stock muerto evitado",
        fmt_units(impact['unsold_saved_units']) + " uds",
        "vs naive baseline",
        delta_positive=True,
    )
with c3:
    kpi_card(
        "α óptimo",
        f"{alpha_optimal:.1f}",
        "peso del score económico",
        delta_positive=False,
    )
with c4:
    kpi_card(
        "Gap vs Oracle",
        fmt_eur(gap_to_oracle),
        "margen de mejora futuro",
        delta_positive=False,
    )

st.markdown("")

# Hero summary
st.markdown(
    f"""
    <div class="hero-box">
    <strong>📌 Resultado clave:</strong> reasignar la producción entre artículos según
    su margen neto esperado (no según volumen) genera <strong>+{impact['margin_uplift_eur']/1e3:.0f}K €</strong>
    de margen adicional <em>al mismo presupuesto de unidades</em> que el baseline humano.
    El optimizador no produce más — produce <em>mejor</em>.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SCENARIO COMPARISON CHART
# ============================================================
st.markdown("### 🎯 Comparativa de 4 escenarios")

# Order from pnl_summary (which has clean scenario names)
order_keys = ['Oracle', 'Naive', 'Forecast', 'Optimized']
ordered = []
for prefix in order_keys:
    for name in pnl['scenario']:
        if name.startswith(prefix):
            ordered.append(name)
            break

color_seq = [COLORS['oracle'], COLORS['naive'], COLORS['forecast'], COLORS['optimized']]
pnl_ord = pnl.set_index('scenario').loc[ordered].reset_index()

col_a, col_b, col_c = st.columns(3)

with col_a:
    fig = go.Figure(go.Bar(
        x=pnl_ord['scenario'],
        y=pnl_ord['total_production'] / 1e6,
        marker_color=color_seq,
        text=[f"{v/1e6:.2f}M" for v in pnl_ord['total_production']],
        textposition='outside',
    ))
    fig.update_layout(
        title="<b>Producción total</b> (M unidades)",
        showlegend=False, height=380,
        margin=dict(t=50, b=120, l=20, r=20),
        plot_bgcolor='white',
        yaxis_title="M unidades",
    )
    fig.update_xaxes(tickangle=25)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    fig = go.Figure(go.Bar(
        x=pnl_ord['scenario'],
        y=pnl_ord['total_unsold'] / 1e3,
        marker_color=color_seq,
        text=[f"{v/1e3:.0f}K" for v in pnl_ord['total_unsold']],
        textposition='outside',
    ))
    fig.update_layout(
        title="<b>Stock no vendido</b> (K unidades)",
        showlegend=False, height=380,
        margin=dict(t=50, b=120, l=20, r=20),
        plot_bgcolor='white',
        yaxis_title="K unidades",
    )
    fig.update_xaxes(tickangle=25)
    st.plotly_chart(fig, use_container_width=True)

with col_c:
    fig = go.Figure(go.Bar(
        x=pnl_ord['scenario'],
        y=pnl_ord['total_margin'] / 1e6,
        marker_color=color_seq,
        text=[f"{v/1e6:.2f}M €" for v in pnl_ord['total_margin']],
        textposition='outside',
    ))
    fig.update_layout(
        title="<b>Margen total</b> (M€)",
        showlegend=False, height=380,
        margin=dict(t=50, b=120, l=20, r=20),
        plot_bgcolor='white',
        yaxis_title="M €",
    )
    fig.update_xaxes(tickangle=25)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ALPHA SENSITIVITY
# ============================================================
st.markdown("### 🎛️ Sensibilidad al parámetro α")
st.markdown(
    "α controla cuánto pesa el score económico (margen neto esperado por unidad) "
    "frente al volumen de demanda predicho. El sweet spot está en α = "
    f"**{alpha_optimal:.1f}**: penalizaciones más agresivas destruyen "
    "valor al desviar producción de productos populares."
)

best_alpha = sens.loc[sens['margin_M'].idxmax()]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=sens['alpha'], y=sens['margin_M'],
    mode='lines+markers', line=dict(color=COLORS['optimized'], width=3),
    marker=dict(size=10), name='Margen total',
))
fig.add_vline(
    x=best_alpha['alpha'], line_dash='dash', line_color=COLORS['naive'],
    annotation_text=f"Óptimo: α={best_alpha['alpha']:.1f} → {best_alpha['margin_M']:.2f} M€",
    annotation_position='top right',
)
fig.update_layout(
    height=420,
    margin=dict(t=20, b=40, l=20, r=20),
    plot_bgcolor='white',
    xaxis_title="α (peso del score económico)",
    yaxis_title="Margen total (M€)",
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# CAVEATS & METHODOLOGY
# ============================================================
with st.expander("📋 Metodología y caveats"):
    st.markdown(f"""
    **Dataset**: H&M Personalized Fashion Recommendations (Kaggle)
    — 31.8M transacciones, 105K artículos, septiembre 2018 – septiembre 2020.

    **Modelos**:
    - Demanda: LightGBM con log-target, validación temporal. WAPE = 0.53 (mejora 6.9% sobre baseline naive).
    - Devoluciones: LightGBM classifier sobre labels semi-sintéticas (tasas sectoriales Narvar/Optoro). ROC-AUC = 0.64.

    **Supuestos económicos** (`src/config.py`):
    - Margen bruto: 53% (H&M annual report)
    - Coste devolución: 18 €/ud (Optoro)
    - Markdown medio: 40%
    - Tasa destrucción stock muerto: 15%

    **Escenarios**:
    - **Oracle (perfect info)**: produce `actual_demand × 1.2`. Upper bound teórico — no comparable de forma realista.
    - **Naive baseline**: forecast humano = demanda real distorsionada con error gaussiano (σ=25%). Simula la decisión de un buyer sin ML.
    - **Forecast-only (ML)**: predicción del modelo de demanda directa, sin lógica de devoluciones.
    - **Optimized (smart, α={alpha_optimal})**: reasigna el mismo presupuesto de unidades del Oracle ponderando por margen neto esperado por artículo.

    **Limitaciones honestas**:
    - Las labels de devolución son semi-sintéticas (H&M no publica devoluciones). El modelo aprende patrones realistas pero está sesgado a las tasas sectoriales asumidas.
    - El modelo económico es estático: no simula sustitución entre artículos, dinámicas de inventario inter-temporal ni elasticidad de precio.
    - El cap de demanda (2.5× predicción) es un parámetro, no derivado de datos.
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:13px;'>"
    "Fashion AI Engine · Portfolio project</div>",
    unsafe_allow_html=True,
)