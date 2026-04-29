"""
Economic model used by the Decision Simulator.

Computes per-article expected margin given:
- predicted demand
- predicted return probability
- production decision (quantity to produce)

All economic constants come from `src.config` and are sourced from
public industry reports (cited in the README).
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

    Returns a dict with the breakdown so the simulator can show it
    transparently in the UI.
    """
    sold_full_price = min(predicted_demand, production_qty)
    unsold = max(production_qty - predicted_demand, 0.0)
    sold_at_markdown = unsold * (1 - destruction_rate)
    destroyed = unsold * destruction_rate

    revenue_full = sold_full_price * unit_price
    revenue_markdown = sold_at_markdown * unit_price * (1 - markdown_pct)

    cost_of_goods = production_qty * unit_price * (1 - gross_margin_pct)
    return_cost = sold_full_price * return_probability * return_cost_eur
    destruction_cost = destroyed * unit_price * (1 - gross_margin_pct)  # sunk

    margin = (
        revenue_full + revenue_markdown
        - cost_of_goods
        - return_cost
        - destruction_cost
    )

    return {
        "production_qty": production_qty,
        "sold_full_price": sold_full_price,
        "sold_at_markdown": sold_at_markdown,
        "destroyed": destroyed,
        "revenue_full": revenue_full,
        "revenue_markdown": revenue_markdown,
        "cost_of_goods": cost_of_goods,
        "return_cost": return_cost,
        "destruction_cost": destruction_cost,
        "expected_margin": margin,
    }


def aggregate_pnl(per_article_results: pd.DataFrame) -> dict[str, float]:
    """
    Aggregate per-article economic results into a portfolio-level P&L.

    Expects a DataFrame with the columns returned by
    `expected_margin_per_article`, one row per article.
    """
    return {
        "total_production": per_article_results["production_qty"].sum(),
        "total_sold_full": per_article_results["sold_full_price"].sum(),
        "total_unsold": (
            per_article_results["sold_at_markdown"].sum()
            + per_article_results["destroyed"].sum()
        ),
        "total_revenue": (
            per_article_results["revenue_full"].sum()
            + per_article_results["revenue_markdown"].sum()
        ),
        "total_return_cost": per_article_results["return_cost"].sum(),
        "total_margin": per_article_results["expected_margin"].sum(),
    }
