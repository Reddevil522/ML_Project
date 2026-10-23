"""Prediction pipeline for Wine Quality Prediction System.

Provides the WineQualityPredictor class to load trained model artifacts,
validate input feature payloads, and compute regression inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd

# Support both module imports and direct script execution
try:
    from .preprocessing import FEATURE_NAMES
    from .train import get_default_model_path
except ImportError:
    from preprocessing import FEATURE_NAMES
    from train import get_default_model_path

logger = logging.getLogger("wine_quality.predict")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Map from underscore field names (used in JSON/FastAPI/Angular) to exact CSV column headers
FEATURE_NAME_MAPPING: dict[str, str] = {
    "fixed_acidity": "fixed acidity",
    "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid",
    "residual_sugar": "residual sugar",
    "chlorides": "chlorides",
    "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide": "total sulfur dioxide",
    "density": "density",
    "pH": "pH",
    "ph": "pH",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
}


class WineQualityPredictor:
    """Predictor service for evaluating red wine quality scores."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        """Initialize the predictor by loading the trained pipeline artifact.

        Args:
            model_path: Optional path to the .joblib model file.
        """
        if model_path is None:
            self.model_path = get_default_model_path()
        else:
            self.model_path = Path(model_path).resolve()

        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load model from disk if available."""
        if not self.model_path.exists():
            logger.warning(
                "Model artifact not found at '%s'. "
                "Predictor initialized in standby mode. Train model using 'python src/train.py'.",
                self.model_path,
            )
            self.model = None
            return

        try:
            self.model = joblib.load(self.model_path)
            logger.info("Successfully loaded model artifact from %s", self.model_path)
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", self.model_path, exc)
            self.model = None
            raise

    @property
    def is_ready(self) -> bool:
        """Check whether the model is loaded and ready for inference."""
        return self.model is not None

    def _normalize_feature_dict(self, data: dict[str, Any]) -> dict[str, float]:
        """Normalize input dictionary keys to standard UCI feature names.

        Supports both standard CSV column names ('fixed acidity')
        and snake_case API keys ('fixed_acidity').
        """
        normalized: dict[str, float] = {}

        for expected_feature in FEATURE_NAMES:
            # Direct match
            if expected_feature in data:
                val = data[expected_feature]
            else:
                # Find matching alias in FEATURE_NAME_MAPPING
                matching_alias = None
                for alias, target in FEATURE_NAME_MAPPING.items():
                    if target == expected_feature and alias in data:
                        matching_alias = alias
                        break
                if matching_alias is not None:
                    val = data[matching_alias]
                else:
                    raise ValueError(
                        f"Missing required feature '{expected_feature}' in input payload."
                    )

            try:
                normalized[expected_feature] = float(val)
            except (ValueError, TypeError) as err:
                raise TypeError(
                    f"Feature '{expected_feature}' value '{val}' cannot be converted to float."
                ) from err

        return normalized

    def prepare_dataframe(
        self, input_data: Union[dict[str, Any], list[dict[str, Any]], pd.DataFrame]
    ) -> pd.DataFrame:
        """Convert input payloads into a validated DataFrame matching FEATURE_NAMES.

        Args:
            input_data: Single dictionary, list of dictionaries, or DataFrame.

        Returns:
            pd.DataFrame: Formatted DataFrame with exactly 11 features in the correct order.
        """
        if isinstance(input_data, dict):
            normalized = self._normalize_feature_dict(input_data)
            return pd.DataFrame([normalized], columns=FEATURE_NAMES)

        if isinstance(input_data, list):
            if not input_data:
                raise ValueError("Input feature list cannot be empty.")
            normalized_list = [self._normalize_feature_dict(item) for item in input_data]
            return pd.DataFrame(normalized_list, columns=FEATURE_NAMES)

        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
            # Rename any snake_case columns if present
            rename_map = {
                k: v for k, v in FEATURE_NAME_MAPPING.items() if k in df.columns and v not in df.columns
            }
            if rename_map:
                df = df.rename(columns=rename_map)

            missing = [f for f in FEATURE_NAMES if f not in df.columns]
            if missing:
                raise ValueError(f"DataFrame is missing required features: {missing}")

            return df[FEATURE_NAMES]

        raise TypeError(
            f"Unsupported input type '{type(input_data)}'. Expected dict, list of dicts, or DataFrame."
        )

    def predict(
        self, input_data: Union[dict[str, Any], list[dict[str, Any]], pd.DataFrame]
    ) -> np.ndarray:
        """Compute regression prediction for given input features.

        Args:
            input_data: Feature payload.

        Returns:
            np.ndarray: Continuous quality predictions.

        Raises:
            RuntimeError: If model has not been trained or loaded.
        """
        if not self.is_ready:
            raise RuntimeError(
                f"Model artifact not loaded from '{self.model_path}'. "
                "Train the model first using 'python src/train.py'."
            )

        features_df = self.prepare_dataframe(input_data)
        predictions = self.model.predict(features_df)
        return np.asarray(predictions, dtype=float)

    def predict_single(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute prediction for a single wine sample.

        Args:
            input_dict: Dictionary containing the 11 wine features.

        Returns:
            dict: Structured prediction response.
        """
        preds = self.predict(input_dict)
        pred_value = float(preds[0])
        rounded = int(round(pred_value))

        return {
            "predicted_quality": round(pred_value, 2),
            "rounded_quality": rounded,
            "status": "success",
            "model_path": str(self.model_path.name),
        }


if __name__ == "__main__":
    predictor = WineQualityPredictor()
    if predictor.is_ready:
        sample_wine = {
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.70,
            "citric_acid": 0.00,
            "residual_sugar": 1.9,
            "chlorides": 0.076,
            "free_sulfur_dioxide": 11.0,
            "total_sulfur_dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4,
        }
        logger.info("Running sample inference with median wine properties...")
        result = predictor.predict_single(sample_wine)
        logger.info("Sample inference result: %s", result)
    else:
        logger.warning("Model not found. Run 'python src/train.py' first.")
