from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fashion_ai_engine.config import load_config
from fashion_ai_engine.features import image_to_features, iter_labeled_images


def train(data_dir: Path, model_path: Path, config_path: Path) -> dict:
    config = load_config(config_path)
    samples = iter_labeled_images(data_dir)

    x = np.vstack(
        [
            image_to_features(
                image_path,
                image_size=config.image_size,
                histogram_bins=config.histogram_bins,
            )
            for image_path, _label in samples
        ]
    )
    y = np.array([label for _image_path, label in samples])

    if len(set(y)) < 2:
        raise ValueError("At least two class folders are required for training.")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )

    class_counts = np.bincount(np.unique(y, return_inverse=True)[1])
    estimated_test_samples = int(np.ceil(len(y) * config.test_size))
    can_holdout = (
        len(y) >= 10
        and min(class_counts) >= 2
        and estimated_test_samples >= len(set(y))
    )

    if can_holdout:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        accuracy = round(float(report["accuracy"]), 4)
    else:
        model.fit(x, y)
        accuracy = None

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return {
        "model_path": str(model_path),
        "samples": len(samples),
        "classes": sorted(str(label) for label in set(y)),
        "accuracy": accuracy,
        "evaluation": "holdout" if can_holdout else "skipped_small_dataset",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fashion image classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-path", type=Path, default=Path("models/fashion_classifier.joblib"))
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    args = parser.parse_args()

    result = train(args.data_dir, args.model_path, args.config)
    print(result)


if __name__ == "__main__":
    main()
