"""Estilos compartidos: paleta SaaS premium, CSS, helpers de formato.

Inspiración: Stripe, PostHog, Linear. Tipografía Inter, mucho whitespace,
KPI cards con sombra sutil, paleta restringida y profesional.
"""
import streamlit as st

# Paleta SaaS premium — verde como acento principal
COLORS = {
    # Acentos por escenario (consistencia con notebooks)
    'oracle':      '#94a3b8',  # gris (techo teórico)
    'naive':       '#ef4444',  # rojo (peor)
    'forecast':    '#f59e0b',  # ámbar (intermedio)
    'optimized':   '#10b981',  # verde (ganador)
    # UI
    'primary':     '#0f172a',  # casi negro (titulares)
    'text':        '#1e293b',  # gris muy oscuro (cuerpo)
    'accent':      '#10b981',  # verde corporativo
    'accent_dark': '#059669',
    'muted':       '#64748b',  # texto secundario
    'border':      '#e2e8f0',  # borders sutiles
    'bg':          '#ffffff',
    'bg_card':     '#ffffff',
    'bg_subtle':   '#f8fafc',  # fondo de zonas destacadas
    'good':        '#10b981',
    'bad':         '#ef4444',
}


def inject_css() -> None:
    """CSS global. Hace que la app no parezca Streamlit por defecto."""
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* === Tipografía global === */
        html, body, [class*="css"], .stApp, .stMarkdown, p, span, div, label {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Inter', sans-serif !important;
            color: {COLORS['primary']} !important;
            letter-spacing: -0.02em !important;
            font-weight: 700 !important;
        }}
        h1 {{ font-size: 2.5rem !important; line-height: 1.1 !important; }}
        h2 {{ font-size: 1.75rem !important; line-height: 1.2 !important; margin-top: 2rem !important; }}
        h3 {{ font-size: 1.25rem !important; line-height: 1.3 !important; margin-top: 1.5rem !important; }}

        /* === Streamlit chrome === */
        #MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] {{
            visibility: hidden;
        }}
        header[data-testid="stHeader"] {{ background: transparent; }}

        /* === Sidebar === */
        section[data-testid="stSidebar"] {{
            background-color: {COLORS['bg_subtle']};
            border-right: 1px solid {COLORS['border']};
        }}
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] li {{
            font-size: 0.9rem;
        }}

        /* === KPI cards (premium) === */
        .kpi-card {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            height: 100%;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            transition: box-shadow 0.2s ease;
        }}
        .kpi-card:hover {{
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        }}
        .kpi-label {{
            font-size: 0.7rem;
            color: {COLORS['muted']};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        .kpi-value {{
            font-size: 2rem;
            font-weight: 700;
            color: {COLORS['primary']};
            line-height: 1.1;
            margin: 0;
        }}
        .kpi-delta-positive {{
            font-size: 0.85rem;
            color: {COLORS['accent']};
            font-weight: 600;
            margin-top: 0.4rem;
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .kpi-delta-negative {{
            font-size: 0.85rem;
            color: {COLORS['bad']};
            font-weight: 600;
            margin-top: 0.4rem;
        }}
        .kpi-delta-neutral {{
            font-size: 0.85rem;
            color: {COLORS['muted']};
            margin-top: 0.4rem;
            font-weight: 500;
        }}

        /* === Hero summary box === */
        .hero-box {{
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
            border: 1px solid #bbf7d0;
            border-left: 3px solid {COLORS['accent']};
            padding: 1.25rem 1.5rem;
            border-radius: 8px;
            margin: 1.5rem 0;
            color: {COLORS['text']};
            line-height: 1.6;
        }}
        .hero-box strong {{ color: {COLORS['primary']}; }}

        /* === Eyebrow text === */
        .eyebrow {{
            font-size: 0.7rem;
            font-weight: 700;
            color: {COLORS['accent']};
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.5rem;
        }}

        /* === Section divider === */
        .section-divider {{
            border: none;
            border-top: 1px solid {COLORS['border']};
            margin: 2.5rem 0 1.5rem 0;
        }}

        /* === Subtle caption === */
        .caption {{
            color: {COLORS['muted']};
            font-size: 0.85rem;
            line-height: 1.55;
            max-width: 720px;
        }}

        /* === Streamlit metric reset (use our cards instead) === */
        [data-testid="stMetricValue"] {{
            font-family: 'Inter', sans-serif !important;
        }}

        /* === Clean expander === */
        .streamlit-expanderHeader {{
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            color: {COLORS['primary']} !important;
        }}

        /* === Plotly chart container === */
        [data-testid="stPlotlyChart"] {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 0.5rem;
        }}
        </style>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = None,
             delta_kind: str = "positive") -> None:
    """KPI card con HTML custom.

    delta_kind: 'positive' (verde) | 'negative' (rojo) | 'neutral' (gris)
    """
    if delta:
        cls = {
            'positive': 'kpi-delta-positive',
            'negative': 'kpi-delta-negative',
            'neutral':  'kpi-delta-neutral',
        }.get(delta_kind, 'kpi-delta-neutral')
        delta_html = f'<div class="{cls}">{delta}</div>'
    else:
        delta_html = ''

    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    """Pequeño label corporativo encima de un h1/h2."""
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def section_divider() -> None:
    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)


def caption(text: str) -> None:
    st.markdown(f'<p class="caption">{text}</p>', unsafe_allow_html=True)


def fmt_eur(v: float, decimals: int = 1) -> str:
    """Formatea un número como €. Ej: 175536 → '+175.5K €'."""
    sign = '+' if v > 0 else ('' if v == 0 else '-')
    v = abs(v)
    if v >= 1_000_000:
        return f"{sign}{v/1e6:.{decimals}f}M €"
    if v >= 1_000:
        return f"{sign}{v/1e3:.{decimals}f}K €"
    return f"{sign}{v:.0f} €"


def fmt_units(v: float) -> str:
    """Formatea unidades. Ej: 5612 → '5.6K'."""
    if abs(v) >= 1_000_000:
        return f"{v/1e6:.2f}M"
    if abs(v) >= 1_000:
        return f"{v/1e3:.1f}K"
    return f"{v:.0f}"


def plotly_layout(fig, height: int = 380):
    """Aplica layout SaaS premium consistente a cualquier gráfico Plotly."""
    fig.update_layout(
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif", size=12, color=COLORS['text']),
        title=dict(
            font=dict(family="Inter, sans-serif", size=14, color=COLORS['primary']),
            x=0, xanchor='left',
        ),
        margin=dict(t=40, b=40, l=20, r=20),
        legend=dict(
            orientation='h', yanchor='bottom', y=-0.25, xanchor='left', x=0,
            font=dict(size=11), bgcolor='rgba(0,0,0,0)',
        ),
        xaxis=dict(
            showgrid=False, linecolor=COLORS['border'],
            tickfont=dict(size=11), title_font=dict(size=11),
        ),
        yaxis=dict(
            gridcolor=COLORS['border'], linecolor=COLORS['border'],
            tickfont=dict(size=11), title_font=dict(size=11),
        ),
    )
    return fig
