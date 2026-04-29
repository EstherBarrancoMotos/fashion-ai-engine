"""Smoke tests for the economic model."""

from src.simulator.economic_model import expected_margin_per_article


def test_margin_breakdown_keys():
    result = expected_margin_per_article(
        predicted_demand=100,
        return_probability=0.2,
        production_qty=120,
        unit_price=29.99,
    )
    expected_keys = {
        "production_qty", "sold_full_price", "sold_at_markdown",
        "destroyed", "revenue_full", "revenue_markdown",
        "cost_of_goods", "return_cost", "destruction_cost",
        "expected_margin",
    }
    assert expected_keys.issubset(result.keys())


def test_overproduction_reduces_margin():
    """Producing way more than expected demand must hurt the bottom line."""
    balanced = expected_margin_per_article(100, 0.2, 100, 29.99)
    overproduced = expected_margin_per_article(100, 0.2, 200, 29.99)
    assert overproduced["expected_margin"] < balanced["expected_margin"]


def test_high_returns_reduce_margin():
    low_return = expected_margin_per_article(100, 0.05, 100, 29.99)
    high_return = expected_margin_per_article(100, 0.40, 100, 29.99)
    assert high_return["expected_margin"] < low_return["expected_margin"]
