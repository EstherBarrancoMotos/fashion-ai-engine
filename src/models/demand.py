"""
Demand prediction model.

Predicts weekly sales per article. The interface (fit / predict / save / load)
is consistent across all models so the simulator can call them generically.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DemandModel:
    """LightGBM-based weekly demand forecaster (placeholder)."""

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DemandModel":
        """Train the model. To be implemented in notebook 02."""
        raise NotImplementedError("Implement in notebook 02_demand_prediction.ipynb")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict weekly demand for the given features."""
        raise NotImplementedError

    def save(self, path: Path | str) -> None:
        """Persist the trained model."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path | str) -> "DemandModel":
        """Load a saved model."""
        raise NotImplementedError
