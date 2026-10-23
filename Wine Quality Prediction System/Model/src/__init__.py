"""Wine Quality Prediction System - ML Backend Package.

This package provides preprocessing, training, evaluation, and inference
pipelines for predicting red wine quality based on physicochemical properties.
"""

from .preprocessing import (
    FEATURE_NAMES,
    TARGET_COLUMN,
    load_data,
    split_data,
    split_features_target,
    create_preprocessor,
    validate_dataset,
    generate_eda_visualizations,
    save_validation_reports,
)
# pyrefly: ignore[missing-import]
from .predict import WineQualityPredictor
# pyrefly: ignore[missing-import]
from .train import train_pipeline
# pyrefly: ignore[missing-import]
from .evaluate import evaluate_model

__version__ = "0.1.0"
__all__ = [
    "FEATURE_NAMES",
    "TARGET_COLUMN",
    "load_data",
    "split_data",
    "split_features_target",
    "create_preprocessor",
    "validate_dataset",
    "generate_eda_visualizations",
    "save_validation_reports",
    "WineQualityPredictor",
    "train_pipeline",
    "evaluate_model",
]
