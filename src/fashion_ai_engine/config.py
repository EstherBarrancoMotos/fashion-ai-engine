from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 224
    histogram_bins: int = 24
    test_size: float = 0.2
    random_state: int = 42
    model_path: Path = Path("models/fashion_classifier.joblib")


def load_config(path: str | Path = "configs/model.yaml") -> ModelConfig:
    config_path = Path(path)
    if not config_path.exists():
        return ModelConfig()

    with config_path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}

    return ModelConfig(
        image_size=int(values.get("image_size", ModelConfig.image_size)),
        histogram_bins=int(values.get("histogram_bins", ModelConfig.histogram_bins)),
        test_size=float(values.get("test_size", ModelConfig.test_size)),
        random_state=int(values.get("random_state", ModelConfig.random_state)),
        model_path=Path(values.get("model_path", ModelConfig.model_path)),
    )
