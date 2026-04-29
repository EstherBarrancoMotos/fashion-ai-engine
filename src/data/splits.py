"""
Temporal splits for time-series ML.

NEVER use random splits for this project. All training, validation and
testing must respect chronological order to avoid information leakage.
"""

from __future__ import annotations

import pandas as pd

from src.config import TEST_END, TRAIN_END, VAL_END


def temporal_split(
    df: pd.DataFrame,
    date_col: str = "t_dat",
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
    test_end: str = TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe chronologically into train / val / test.

    Returns
    -------
    train, val, test : pd.DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    train = df[df[date_col] <= train_end]
    val = df[(df[date_col] > train_end) & (df[date_col] <= val_end)]
    test = df[(df[date_col] > val_end) & (df[date_col] <= test_end)]

    return train, val, test
