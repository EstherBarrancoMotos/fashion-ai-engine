"""
Economic model used by the Decision Simulator (v2 — with lost demand penalty).

Key fix vs v1: when production < demand, we now account for the
opportunity cost of unserved customers. Without this, the simulator
incorrectly recommends massive production cuts because it only sees
the savings, never the lost revenue.

References for assumptions:
- Gross margin: H&M Group annual report 2023 (~52-55%)
- Return handling cost: Optoro Returns Industry Report
- Markdown: standard fashion retail benchmarks
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    GROSS_MARGIN_PCT,
    MARKDOWN_DISCOUNT_PCT,
    RETURN_HANDLING_COST_EUR,
    UNSOLD_DESTRUCTION_RATE,
)


def expected_margin_per_article(
    predicted_demand: float,
    return_probability: float,
    production_qty: float,
    unit_price: float,
    gross_margin_pct: float = GROSS_MARGIN_PCT,
    return_cost_eur: float = RETURN_HANDLING_COST_EUR,
    markdown_pct: float = MARKDOWN_DISCOUNT_PCT,
    destruction_rate: float = UNSOLD_DESTRUCTION_RATE,
) -> dict[str, float]:
    """
    Estimate the expected margin (in EUR) for a single article given
    a production decision.

    The model accounts for:
    - Revenue from full-price sales
    - Revenue from markdown sales of unsold stock
    - Cost of goods produced
    - Cost of handling returns (Optoro benchmark)
    - Cost of destroyed unsold stock
    - Lost margin from unserved demand (NEW in v2)

    Returns a breakdown so the simulator can show it transparently.
    """
    sold_full_price = min(predicted_demand, production_qty)
    unsold = max(production_qty - predicted_demand, 0.0)
    sold_at_markdown = unsold * (1 - destruction_rate)
    destroyed = unsold * destruction_rate

    # Lost demand: cuando produces menos que la demanda real
    lost_demand = max(predicted_demand - production_qty, 0.0)
    lost_margin_eur = lost_demand * unit_price * gross_margin_pct

    revenue_full = sold_full_price * unit_price
    revenue_markdown = sold_at_markdown * unit_price * (1 - markdown_pct)

    cost_of_goods = production_qty * unit_price * (1 - gross_margin_pct)
    return_cost = sold_full_price * return_probability * return_cost_eur
    destruction_cost = destroyed * unit_price * (1 - gross_margin_pct)

    margin = (
        revenue_full + revenue_markdown
        - cost_of_goods
        - return_cost
        - destruction_cost
        - lost_margin_eur     # ← NUEVO: penaliza demanda no satisfecha
    )

    return {
        "production_qty": production_qty,
        "sold_full_price": sold_full_price,
        "sold_at_markdown": sold_at_markdown,
        "destroyed": destroyed,
        "lost_demand": lost_demand,                # ← NUEVO
        "revenue_full": revenue_full,
        "revenue_markdown": revenue_markdown,
        "cost_of_goods": cost_of_goods,
        "return_cost": return_cost,
        "destruction_cost": destruction_cost,
        "lost_margin_cost": lost_margin_eur,       # ← NUEVO
        "expected_margin": margin,
    }


def aggregate_pnl(per_article_results: pd.DataFrame) -> dict[str, float]:
    """Aggregate per-article results into a portfolio-level P&L."""
    return {
        "total_production": per_article_results["production_qty"].sum(),
        "total_sold_full": per_article_results["sold_full_price"].sum(),
        "total_unsold": (
            per_article_results["sold_at_markdown"].sum()
            + per_article_results["destroyed"].sum()
        ),
        "total_lost_demand": per_article_results["lost_demand"].sum(),
        "total_revenue": (
            per_article_results["revenue_full"].sum()
            + per_article_results["revenue_markdown"].sum()
        ),
        "total_return_cost": per_article_results["return_cost"].sum(),
        "total_lost_margin_cost": per_article_results["lost_margin_cost"].sum(),
        "total_margin": per_article_results["expected_margin"].sum(),
    }
