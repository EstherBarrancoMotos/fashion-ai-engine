"""Estilos compartidos: paleta, CSS, helpers de formato."""
import streamlit as st

# Paleta consistente con los gráficos del notebook
COLORS = {
    'oracle':      '#94a3b8',  # gris (techo teórico)
    'naive':       '#ef4444',  # rojo (peor caso)
    'forecast':    '#f59e0b',  # naranja (intermedio)
    'optimized':   '#10b981',  # verde (ganador)
    'primary':     '#0f172a',
    'accent':      '#10b981',
    'muted':       '#64748b',
    'bg_card':     '#f8fafc',
    'border':      '#e2e8f0',
}


def inject_css():
    """CSS global para que la app no parezca Streamlit por defecto."""
    st.markdown(f"""
        <style>
        /* Tipografía */
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
        }}
        /* KPI cards */
        .kpi-card {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 20px 24px;
            height: 100%;
        }}
        .kpi-label {{
            font-size: 13px;
            color: {COLORS['muted']};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 32px;
            font-weight: 700;
            color: {COLORS['primary']};
            line-height: 1.1;
            margin: 0;
        }}
        .kpi-delta-positive {{
            font-size: 14px;
            color: {COLORS['accent']};
            font-weight: 600;
            margin-top: 6px;
        }}
        .kpi-delta-neutral {{
            font-size: 14px;
            color: {COLORS['muted']};
            margin-top: 6px;
        }}
        /* Hero box */
        .hero-box {{
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
            border-left: 4px solid {COLORS['accent']};
            padding: 20px 24px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        /* Hide streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        </style>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = None, delta_positive: bool = True):
    """Renderiza una KPI card con HTML custom."""
    delta_class = 'kpi-delta-positive' if delta_positive else 'kpi-delta-neutral'
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ''
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def fmt_eur(v: float, decimals: int = 0) -> str:
    """Formatea un número como €. Ej: 175536 → '+175.5K €'."""
    sign = '+' if v > 0 else ('' if v == 0 else '-')
    v = abs(v)
    if v >= 1_000_000:
        return f"{sign}{v/1e6:.{decimals+1}f}M €"
    if v >= 1_000:
        return f"{sign}{v/1e3:.{decimals+1}f}K €"
    return f"{sign}{v:.0f} €"


def fmt_units(v: float) -> str:
    """Formatea unidades. Ej: 505538 → '505.5K'."""
    if abs(v) >= 1_000_000:
        return f"{v/1e6:.2f}M"
    if abs(v) >= 1_000:
        return f"{v/1e3:.1f}K"
    return f"{v:.0f}"