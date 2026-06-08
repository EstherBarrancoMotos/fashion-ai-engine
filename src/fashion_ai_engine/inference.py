from __future__ import annotations

import os
from pathlib import Path

import joblib

from fashion_ai_engine.config import ModelConfig, load_config
from fashion_ai_engine.features import image_metadata, image_to_features


class FashionPredictor:
    def __init__(self, config: ModelConfig, model_path: Path | None = None) -> None:
        self.config = config
        self.model_path = model_path or Path(os.getenv("MODEL_PATH", config.model_path))
        self.model = joblib.load(self.model_path) if self.model_path.exists() else None

    @classmethod
    def from_environment(cls) -> "FashionPredictor":
        config_path = os.getenv("MODEL_CONFIG", "configs/model.yaml")
        return cls(load_config(config_path))

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image_bytes: bytes) -> dict:
        metadata = image_metadata(image_bytes, image_size=self.config.image_size)

        if self.model is None:
            return {
                "model_loaded": False,
                "prediction": None,
                "confidence": None,
                "message": "No trained model found. Train one with python -m fashion_ai_engine.training.",
                "image": metadata,
            }

        features = image_to_features(
            image_bytes,
            image_size=self.config.image_size,
            histogram_bins=self.config.histogram_bins,
        ).reshape(1, -1)

        label = self.model.predict(features)[0]
        confidence = None
        probabilities = {}

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            classes = list(self.model.classes_)
            probabilities = {
                str(class_name): round(float(probability), 4)
                for class_name, probability in zip(classes, proba)
            }
            confidence = max(probabilities.values()) if probabilities else None

        return {
            "model_loaded": True,
            "prediction": str(label),
            "confidence": confidence,
            "probabilities": probabilities,
            "image": metadata,
        }
