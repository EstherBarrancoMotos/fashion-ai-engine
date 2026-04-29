"""
Return probability prediction model.

Predicts the probability that a given article will be returned.
Trained on synthetic-but-realistic labels derived from industry benchmarks
(see config.RETURN_RATES_BY_CATEGORY) and modulated by observable features.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ReturnModel:
    """Gradient boosting classifier for return probability (placeholder)."""

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ReturnModel":
        raise NotImplementedError("Implement in notebook 03_return_prediction.ipynb")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return P(return=1) per row."""
        raise NotImplementedError

    def save(self, path: Path | str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path | str) -> "ReturnModel":
        raise NotImplementedError
