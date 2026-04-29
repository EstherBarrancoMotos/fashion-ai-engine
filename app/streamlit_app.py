"""
Fashion AI Engine — Streamlit App

Entry point. Renders the executive dashboard. Other pages live in app/pages/
and are picked up automatically by Streamlit's multi-page routing.

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Fashion AI Engine",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("👗 Fashion AI Engine")
st.subheader("Smarter production decisions for fashion brands")

st.markdown(
    """
    **Fashion AI Engine** helps fashion brands reduce overproduction,
    returns and unsold inventory by combining:

    1. **Demand prediction** — how much of each product will sell
    2. **Return prediction** — which products are most likely to be returned
    3. **Style intelligence** — which colors, categories and price points work
    4. **Decision simulator** — quantify the margin impact of every decision

    Use the sidebar to navigate between modules.
    """
)

st.info(
    "🚧 This app is under construction. As you train each model in the notebooks, "
    "the corresponding page will come alive with real predictions."
)

# Placeholder KPIs — will be replaced with real numbers once the simulator runs.
st.divider()
st.markdown("### Projected business impact (placeholder)")

col1, col2, col3 = st.columns(3)
col1.metric("Excess inventory reduction", "—", help="Will be computed by the simulator")
col2.metric("Returns avoided", "—")
col3.metric("Estimated margin uplift", "—")
