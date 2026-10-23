"""Model training pipeline for Wine Quality Prediction System.

Trains and systematically compares three regression algorithms:
1. LinearRegression (baseline)
2. DecisionTreeRegressor
3. RandomForestRegressor

Performs 5-fold cross-validation on the isolated training set using Scikit-Learn
pipelines, selects the optimal model based on Mean Absolute Error (MAE) with a
deterministic tie-breaking rule, evaluates the winning model on the holdout test set,
and persists model artifacts, comprehensive metadata, and comparison reports.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Support both module imports (python -m src.train) and direct script execution (python src/train.py)
try:
    from .preprocessing import (
        FEATURE_NAMES,
        TARGET_COLUMN,
        get_default_data_path,
        load_data,
        split_data,
        split_features_target,
    )
except ImportError:
    from preprocessing import (
        FEATURE_NAMES,
        TARGET_COLUMN,
        get_default_data_path,
        load_data,
        split_data,
        split_features_target,
    )

logger = logging.getLogger("wine_quality.train")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_default_model_path() -> Path:
    """Return default path to the saved model artifact.

    Returns:
        Path: ml-backend/models/wine_quality_model.joblib
    """
    backend_dir = Path(__file__).resolve().parent.parent
    models_dir = backend_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir / "wine_quality_model.joblib"


def get_default_metadata_path() -> Path:
    """Return default path to the model metadata JSON file.

    Returns:
        Path: ml-backend/models/wine_quality_model_metadata.json
    """
    backend_dir = Path(__file__).resolve().parent.parent
    models_dir = backend_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir / "wine_quality_model_metadata.json"


def get_default_reports_path() -> Path:
    """Return default path to the training reports directory.

    Returns:
        Path: ml-backend/reports/
    """
    backend_dir = Path(__file__).resolve().parent.parent
    reports_dir = backend_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def build_pipeline(
    model_type: str = "random_forest",
    random_state: int = 42,
) -> Pipeline:
    """Construct an end-to-end scikit-learn pipeline for a specified regressor.

    Combines SimpleImputer (median fallback) and StandardScaler for leak-free
    preprocessing, followed by the regression estimator.

    Args:
        model_type: One of 'linear_regression', 'decision_tree', or 'random_forest'.
            Defaults to 'random_forest'.
        random_state: Seed for deterministic model initialization.

    Returns:
        Pipeline: Unfitted scikit-learn pipeline.

    Raises:
        ValueError: If model_type is not recognized.
    """
    model_type_norm = model_type.lower().replace("_", "").replace(" ", "").strip()

    if model_type_norm in ["linearregression", "linear", "lr"]:
        regressor = LinearRegression()
    elif model_type_norm in ["decisiontreeregressor", "decisiontree", "tree", "dt"]:
        regressor = DecisionTreeRegressor(
            random_state=random_state,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
        )
    elif model_type_norm in ["randomforestregressor", "randomforest", "rf", "forest"]:
        regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            "Supported algorithms: 'linear_regression', 'decision_tree', 'random_forest'."
        )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )
    return pipeline


def get_candidate_models(random_state: int = 42) -> Dict[str, Pipeline]:
    """Instantiate pipelines for all three candidate regression algorithms.

    Args:
        random_state: Random seed for reproducible initialization.

    Returns:
        Dict[str, Pipeline]: Dictionary mapping model names to configured pipelines.
    """
    return {
        "LinearRegression": build_pipeline("linear_regression", random_state=random_state),
        "DecisionTreeRegressor": build_pipeline("decision_tree", random_state=random_state),
        "RandomForestRegressor": build_pipeline("random_forest", random_state=random_state),
    }


def evaluate_candidates_cv(
    candidate_models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Perform identical 5-fold cross-validation across all candidate regression algorithms.

    Evaluates MAE, RMSE, MSE, R², and fit/scoring times strictly on the training partition.

    Args:
        candidate_models: Dictionary of algorithm names and pipeline instances.
        X_train: Training features DataFrame.
        y_train: Training continuous quality targets.
        n_splits: Number of cross-validation folds. Defaults to 5.
        random_state: Random state for KFold shuffling.

    Returns:
        Dict[str, Dict[str, Any]]: Detailed cross-validation results per algorithm.
    """
    logger.info(
        "Starting %d-fold cross-validation on %d training samples...",
        n_splits,
        len(X_train),
    )

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }

    cv_results: Dict[str, Dict[str, Any]] = {}

    for name, pipeline in candidate_models.items():
        logger.info("Evaluating algorithm via CV: %s...", name)
        cv_output = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        mae_scores = -cv_output["test_mae"]
        rmse_scores = -cv_output["test_rmse"]
        mse_scores = -cv_output["test_mse"]
        r2_scores = cv_output["test_r2"]
        fit_times = cv_output["fit_time"]
        score_times = cv_output["score_time"]

        summary = {
            "algorithm": name,
            "cv_mae_mean": round(float(mae_scores.mean()), 4),
            "cv_mae_std": round(float(mae_scores.std()), 4),
            "cv_rmse_mean": round(float(rmse_scores.mean()), 4),
            "cv_rmse_std": round(float(rmse_scores.std()), 4),
            "cv_mse_mean": round(float(mse_scores.mean()), 4),
            "cv_mse_std": round(float(mse_scores.std()), 4),
            "cv_r2_mean": round(float(r2_scores.mean()), 4),
            "cv_r2_std": round(float(r2_scores.std()), 4),
            "avg_fit_time_seconds": round(float(fit_times.mean()), 4),
            "avg_score_time_seconds": round(float(score_times.mean()), 4),
            "fold_mae_scores": [round(float(s), 4) for s in mae_scores],
            "fold_rmse_scores": [round(float(s), 4) for s in rmse_scores],
            "fold_r2_scores": [round(float(s), 4) for s in r2_scores],
        }

        cv_results[name] = summary
        logger.info(
            "  %s -> CV MAE: %.4f (±%.4f) | CV RMSE: %.4f | CV R²: %.4f | Fit Time: %.4fs",
            name,
            summary["cv_mae_mean"],
            summary["cv_mae_std"],
            summary["cv_rmse_mean"],
            summary["cv_r2_mean"],
            summary["avg_fit_time_seconds"],
        )

    return cv_results


def select_best_model(
    cv_results: Dict[str, Dict[str, Any]],
    tolerance: float = 1e-4,
) -> Tuple[str, Dict[str, Any]]:
    """Select the optimal model using CV MAE as the primary metric with deterministic tie-breaking.

    Tie-breaking hierarchy:
    1. Primary: Lowest CV MAE (Mean Absolute Error).
    2. Secondary: Lowest CV RMSE (Root Mean Squared Error).
    3. Tertiary: Highest CV R² (Coefficient of Determination).
    4. Quaternary: Lowest average fit time (computational efficiency).
    5. Quinary: Alphabetical order of algorithm name (guarantees strict determinism).

    Args:
        cv_results: Dictionary of model CV evaluation dictionaries.
        tolerance: Numeric epsilon below which two MAE values are deemed tied.

    Returns:
        Tuple[str, Dict[str, Any]]: (winning_model_name, winning_model_cv_summary).
    """

    def sorting_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[float, float, float, float, str]:
        name, res = item
        # Round to tolerance for tie-breaking
        mae_key = round(res["cv_mae_mean"] / tolerance) * tolerance
        rmse_key = res["cv_rmse_mean"]
        neg_r2_key = -res["cv_r2_mean"]  # higher R² is better, so negate
        fit_time_key = res["avg_fit_time_seconds"]
        return (mae_key, rmse_key, neg_r2_key, fit_time_key, name)

    sorted_candidates = sorted(cv_results.items(), key=sorting_key)
    winner_name, winner_data = sorted_candidates[0]

    logger.info(
        "Deterministic model selection complete. Winner: '%s' (CV MAE: %.4f, CV RMSE: %.4f, CV R²: %.4f)",
        winner_name,
        winner_data["cv_mae_mean"],
        winner_data["cv_rmse_mean"],
        winner_data["cv_r2_mean"],
    )
    return winner_name, winner_data


def validate_saved_artifact(
    model_path: Path,
    expected_features: List[str] = FEATURE_NAMES,
) -> Dict[str, Any]:
    """Verify that the persisted model artifact reloads and predicts accurately on valid DataFrames.

    Args:
        model_path: Path to serialized .joblib file.
        expected_features: List of required feature names in correct order.

    Returns:
        Dict[str, Any]: Validation verification status.

    Raises:
        FileNotFoundError: If artifact is missing.
        RuntimeError: If artifact fails reloading or inference.
    """
    logger.info("Validating reloaded model artifact from: %s", model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at '{model_path}'.")

    try:
        loaded_pipeline = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to reload model artifact with joblib: {exc}") from exc

    # Construct test sample with exact feature columns
    test_sample = pd.DataFrame(
        [
            {
                "fixed acidity": 7.4,
                "volatile acidity": 0.70,
                "citric acid": 0.00,
                "residual sugar": 1.9,
                "chlorides": 0.076,
                "free sulfur dioxide": 11.0,
                "total sulfur dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
            }
        ]
    )[expected_features]

    try:
        pred = loaded_pipeline.predict(test_sample)
        pred_val = float(pred[0])
    except Exception as exc:
        raise RuntimeError(f"Reloaded model failed prediction validation: {exc}") from exc

    if np.isnan(pred_val) or np.isinf(pred_val):
        raise RuntimeError(f"Reloaded model produced invalid numeric output: {pred_val}")

    logger.info("Model reload validation succeeded. Sample test prediction: %.4f", pred_val)
    return {
        "status": "valid",
        "sample_prediction": round(pred_val, 4),
        "steps": [step[0] for step in loaded_pipeline.steps],
    }


def train_pipeline(
    data_path: Path | str | None = None,
    model_output_path: Path | str | None = None,
    reports_dir: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """Execute end-to-end multi-algorithm training, selection, holdout evaluation, and artifact saving.

    Workflow:
    1. Ingest dataset using strict semicolon delimiter.
    2. Separate X and y, preserving exact 11 feature names and order.
    3. Split into 80% train and 20% test partitions using fixed random seed (42).
    4. Run 5-fold cross-validation across LinearRegression, DecisionTreeRegressor,
       and RandomForestRegressor on X_train only (test set isolated).
    5. Deterministically select the winning model based on lowest CV MAE.
    6. Fit winning pipeline on the full X_train.
    7. Evaluate winning pipeline on holdout test set X_test.
    8. Persist .joblib model, metadata JSON, and CV comparison CSV/JSON reports.
    9. Perform validation test on reloaded artifact.

    Args:
        data_path: Optional path to winequality-red.csv.
        model_output_path: Destination path for .joblib model file.
        reports_dir: Destination directory for reports.
        test_size: Proportion of holdout test data. Defaults to 0.2.
        random_state: Random state seed for reproducibility. Defaults to 42.
        cv_folds: Number of cross-validation folds. Defaults to 5.

    Returns:
        Dict[str, Any]: Summary dictionary containing paths, winning model, CV and test metrics.
    """
    start_time = datetime.now(timezone.utc)
    logger.info("Initializing multi-algorithm training pipeline...")

    # Resolve destination paths
    if model_output_path is None:
        target_model_path = get_default_model_path()
    else:
        target_model_path = Path(model_output_path).resolve()

    if reports_dir is None:
        target_reports_dir = get_default_reports_path()
    else:
        target_reports_dir = Path(reports_dir).resolve()

    target_model_path.parent.mkdir(parents=True, exist_ok=True)
    target_reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest data and split features/target
    resolved_data_path = Path(data_path).resolve() if data_path else get_default_data_path()
    raw_df = load_data(filepath=resolved_data_path, sep=";")
    X, y = split_features_target(raw_df, target_column=TARGET_COLUMN, feature_names=FEATURE_NAMES)

    # 2. Split dataset with fixed random seed (test set isolated)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Instantiate candidate algorithms
    candidates = get_candidate_models(random_state=random_state)

    # 4. Cross-validate candidate models strictly on X_train
    cv_results = evaluate_candidates_cv(
        candidate_models=candidates,
        X_train=X_train,
        y_train=y_train,
        n_splits=cv_folds,
        random_state=random_state,
    )

    # 5. Select best model based on CV MAE
    winner_name, winner_cv_data = select_best_model(cv_results)

    # 6. Fit the winning pipeline on full training set
    logger.info("Fitting winning pipeline ('%s') on full training set (%d samples)...", winner_name, len(X_train))
    winning_pipeline = build_pipeline(
        model_type=winner_name,
        random_state=random_state,
    )
    winning_pipeline.fit(X_train, y_train)

    # 7. Evaluate winning pipeline on holdout test set (unseen data)
    logger.info("Evaluating winning pipeline on isolated holdout test split (%d samples)...", len(X_test))
    y_pred = winning_pipeline.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(mse ** 0.5)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    evs = float(explained_variance_score(y_test, y_pred))

    test_metrics = {
        "dataset_rows": int(len(raw_df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "selected_model": winner_name,
        "mean_squared_error": round(mse, 4),
        "root_mean_squared_error": round(rmse, 4),
        "mean_absolute_error": round(mae, 4),
        "r2_score": round(r2, 4),
        "explained_variance_score": round(evs, 4),
        "model_type": str(winning_pipeline),
        "random_state": random_state,
    }

    logger.info("=== Holdout Test Performance (%s) ===", winner_name)
    logger.info("  Holdout MAE  : %.4f", mae)
    logger.info("  Holdout RMSE : %.4f", rmse)
    logger.info("  Holdout MSE  : %.4f", mse)
    logger.info("  Holdout R²   : %.4f", r2)

    # 8. Persist the trained model artifact
    logger.info("Persisting selected model artifact to: %s", target_model_path)
    joblib.dump(winning_pipeline, target_model_path)

    # 9. Persist model metadata JSON
    metadata_path = target_model_path.parent / f"{target_model_path.stem}_metadata.json"
    metadata = {
        "model_name": winner_name,
        "pipeline_steps": [step[0] for step in winning_pipeline.steps],
        "training_timestamp_utc": start_time.isoformat(),
        "target_column": TARGET_COLUMN,
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "random_seed": random_state,
        "dataset_information": {
            "source_path": str(resolved_data_path),
            "total_rows": int(len(raw_df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "test_size": test_size,
        },
        "training_configuration": {
            "cv_folds": cv_folds,
            "primary_selection_metric": "mean_absolute_error (MAE, lower is better)",
            "tie_breaking_hierarchy": [
                "CV MAE (lowest)",
                "CV RMSE (lowest)",
                "CV R² (highest)",
                "Avg Fit Time (lowest)",
                "Alphabetical Algorithm Name",
            ],
            "estimator_hyperparameters": str(winning_pipeline.named_steps["regressor"].get_params()),
        },
        "cross_validation_results": winner_cv_data,
        "holdout_test_evaluation": test_metrics,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Persisted model metadata JSON to: %s", metadata_path)

    # 10. Persist cross-validation comparison reports (CSV and JSON)
    cv_comparison_json_path = target_reports_dir / "cv_model_comparison.json"
    with open(cv_comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)
    logger.info("Persisted CV comparison JSON to: %s", cv_comparison_json_path)

    comparison_records = []
    for model_name, res in cv_results.items():
        comparison_records.append({
            "algorithm": model_name,
            "cv_mae_mean": res["cv_mae_mean"],
            "cv_mae_std": res["cv_mae_std"],
            "cv_rmse_mean": res["cv_rmse_mean"],
            "cv_rmse_std": res["cv_rmse_std"],
            "cv_r2_mean": res["cv_r2_mean"],
            "cv_r2_std": res["cv_r2_std"],
            "avg_fit_time_seconds": res["avg_fit_time_seconds"],
            "selected_as_winner": (model_name == winner_name),
        })
    cv_df = pd.DataFrame(comparison_records)
    cv_comparison_csv_path = target_reports_dir / "cv_model_comparison.csv"
    cv_df.to_csv(cv_comparison_csv_path, index=False)
    logger.info("Persisted CV comparison CSV to: %s", cv_comparison_csv_path)

    # Also persist train_metrics.json for backward compatibility with existing tests
    train_metrics_path = target_reports_dir / "train_metrics.json"
    with open(train_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("Persisted backward-compatible train_metrics.json to: %s", train_metrics_path)

    # 11. Validate reloaded artifact
    validation_status = validate_saved_artifact(target_model_path, expected_features=FEATURE_NAMES)

    return {
        "model_path": str(target_model_path),
        "metadata_path": str(metadata_path),
        "metrics_path": str(train_metrics_path),
        "cv_comparison_json": str(cv_comparison_json_path),
        "cv_comparison_csv": str(cv_comparison_csv_path),
        "selected_model": winner_name,
        "metrics": test_metrics,
        "cv_results": cv_results,
        "validation": validation_status,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Executing model training workflow...")
    training_summary = train_pipeline()

    print("\n" + "=" * 75)
    print("WINE QUALITY MODEL TRAINING & SELECTION REPORT")
    print("=" * 75)
    print(f"Dataset Size:           {training_summary['metrics']['dataset_rows']} samples (Train: {training_summary['metrics']['train_samples']}, Test: {training_summary['metrics']['test_samples']})")
    print("\nCandidate Algorithm Cross-Validation Performance (5 Folds):")
    print("-" * 75)
    print(f"{'Algorithm':<24} | {'CV MAE':<14} | {'CV RMSE':<14} | {'CV R²':<10} | {'Fit Time'}")
    print("-" * 75)
    for name, res in training_summary["cv_results"].items():
        is_winner = " [SELECTED]" if name == training_summary["selected_model"] else ""
        print(
            f"{name + is_winner:<24} | "
            f"{res['cv_mae_mean']:.4f} (±{res['cv_mae_std']:.3f}) | "
            f"{res['cv_rmse_mean']:.4f} (±{res['cv_rmse_std']:.3f}) | "
            f"{res['cv_r2_mean']:7.4f}  | "
            f"{res['avg_fit_time_seconds']:.4f}s"
        )
    print("-" * 75)
    print(f"\nWinning Model:           {training_summary['selected_model']}")
    print(f"Selection Criterion:     Lowest CV Mean Absolute Error (MAE)")
    print(f"\nHoldout Test Set Performance (Unseen 320 Samples):")
    print(f"  - Test MAE:            {training_summary['metrics']['mean_absolute_error']:.4f}")
    print(f"  - Test RMSE:           {training_summary['metrics']['root_mean_squared_error']:.4f}")
    print(f"  - Test MSE:            {training_summary['metrics']['mean_squared_error']:.4f}")
    print(f"  - Test R²:             {training_summary['metrics']['r2_score']:.4f}")
    print(f"  - Explained Variance:  {training_summary['metrics']['explained_variance_score']:.4f}")

    print("\nPersisted Artifacts:")
    print(f"  - Model Pipeline:      {training_summary['model_path']}")
    print(f"  - Metadata JSON:       {training_summary['metadata_path']}")
    print(f"  - CV Comparison JSON:  {training_summary['cv_comparison_json']}")
    print(f"  - CV Comparison CSV:   {training_summary['cv_comparison_csv']}")
    print(f"  - Test Metrics JSON:   {training_summary['metrics_path']}")

    print(f"\nArtifact Reload Validation: {training_summary['validation']['status'].upper()}")
    print(f"Sample Inference Score:     {training_summary['validation']['sample_prediction']}")
    print("=" * 75)
