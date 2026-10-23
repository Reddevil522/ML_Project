"""Unit tests for training, evaluation, and inference pipelines."""

from pathlib import Path
import pytest
import numpy as np

from src.preprocessing import FEATURE_NAMES, load_data, split_features_target, split_data
from src.train import build_pipeline, train_pipeline, get_default_model_path
from src.evaluate import evaluate_model, tune_hyperparameters, tune_single_model
from src.predict import WineQualityPredictor


def test_build_pipeline() -> None:
    """Verify that build_pipeline returns a valid scikit-learn Pipeline."""
    pipe = build_pipeline(random_state=42)
    assert hasattr(pipe, "fit")
    assert hasattr(pipe, "predict")
    assert "scaler" in pipe.named_steps
    assert "regressor" in pipe.named_steps


def test_train_pipeline_execution(tmp_path: Path) -> None:
    """Verify that train_pipeline runs successfully and outputs expected files."""
    model_output = tmp_path / "test_model.joblib"
    reports_output = tmp_path / "reports"

    result = train_pipeline(
        model_output_path=model_output,
        reports_dir=reports_output,
        test_size=0.2,
        random_state=42,
    )

    assert model_output.exists()
    assert (reports_output / "train_metrics.json").exists()
    assert "mean_squared_error" in result["metrics"]
    assert "r2_score" in result["metrics"]
    assert result["metrics"]["test_samples"] == 320


def test_predictor_single_and_batch() -> None:
    """Verify that WineQualityPredictor computes accurate predictions."""
    predictor = WineQualityPredictor()
    assert predictor.is_ready

    sample_dict = {
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

    result = predictor.predict_single(sample_dict)
    assert result["status"] == "success"
    assert isinstance(result["predicted_quality"], float)
    assert 3.0 <= result["predicted_quality"] <= 8.5
    assert isinstance(result["rounded_quality"], int)

    # Test batch predictions
    batch_data = [sample_dict, sample_dict]
    batch_preds = predictor.predict(batch_data)
    assert isinstance(batch_preds, np.ndarray)
    assert len(batch_preds) == 2


def test_predictor_missing_feature_error() -> None:
    """Verify that predictor raises ValueError when required features are omitted."""
    predictor = WineQualityPredictor()
    incomplete_dict = {
        "fixed_acidity": 7.4,
        "alcohol": 9.4,
    }
    with pytest.raises(ValueError) as exc:
        predictor.predict(incomplete_dict)
    assert "Missing required feature" in str(exc.value)


def test_evaluate_model_execution(tmp_path: Path) -> None:
    """Verify that evaluate_model generates metrics, all diagnostic plots, and report."""
    reports_dir = tmp_path / "eval_reports"
    metrics = evaluate_model(reports_dir=reports_dir)

    assert (reports_dir / "metrics.json").exists()
    assert (reports_dir / "evaluation_summary.json").exists()
    assert (reports_dir / "evaluation_report.md").exists()
    assert (reports_dir / "actual_vs_predicted.png").exists()
    assert (reports_dir / "residuals_distribution.png").exists()
    assert (reports_dir / "residuals_vs_predicted.png").exists()
    assert (reports_dir / "feature_importance.png").exists()
    assert (reports_dir / "residuals_plot.png").exists()

    # All 4 core regression metrics required by specification
    assert "mean_absolute_error" in metrics
    assert "mean_squared_error" in metrics
    assert "root_mean_squared_error" in metrics
    assert "r2_score" in metrics
    assert 0.0 < metrics["mean_absolute_error"] < 1.0
    assert 0.0 < metrics["root_mean_squared_error"] < 1.0
    assert 0.0 < metrics["r2_score"] < 1.0


def test_build_pipeline_all_models() -> None:
    """Verify build_pipeline constructs valid pipelines for all three algorithms."""
    for model_name in ["linear_regression", "decision_tree", "random_forest"]:
        pipe = build_pipeline(model_type=model_name, random_state=42)
        assert hasattr(pipe, "fit")
        assert hasattr(pipe, "predict")
        assert "scaler" in pipe.named_steps
        assert "regressor" in pipe.named_steps


def test_deterministic_model_selection() -> None:
    """Verify select_best_model deterministically picks lowest MAE with tie-breaking."""
    from src.train import select_best_model

    mock_cv_results = {
        "ModelA": {
            "cv_mae_mean": 0.5200,
            "cv_rmse_mean": 0.6500,
            "cv_r2_mean": 0.3000,
            "avg_fit_time_seconds": 0.05,
        },
        "ModelB": {
            "cv_mae_mean": 0.4300,
            "cv_rmse_mean": 0.6000,
            "cv_r2_mean": 0.4500,
            "avg_fit_time_seconds": 0.50,
        },
        "ModelC": {
            "cv_mae_mean": 0.4700,
            "cv_rmse_mean": 0.8000,
            "cv_r2_mean": 0.0000,
            "avg_fit_time_seconds": 0.02,
        },
    }

    winner_name, winner_data = select_best_model(mock_cv_results)
    assert winner_name == "ModelB"
    assert winner_data["cv_mae_mean"] == 0.4300


def test_train_pipeline_artifacts_and_metadata(tmp_path: Path) -> None:
    """Verify train_pipeline generates model, metadata JSON, and CV reports."""
    model_output = tmp_path / "custom_wine_model.joblib"
    reports_output = tmp_path / "custom_reports"

    result = train_pipeline(
        model_output_path=model_output,
        reports_dir=reports_output,
        test_size=0.2,
        random_state=42,
        cv_folds=3,  # Fast 3-fold CV for test
    )

    metadata_file = tmp_path / "custom_wine_model_metadata.json"
    cv_json = reports_output / "cv_model_comparison.json"
    cv_csv = reports_output / "cv_model_comparison.csv"

    assert model_output.exists()
    assert metadata_file.exists()
    assert cv_json.exists()
    assert cv_csv.exists()
    assert result["selected_model"] in ["RandomForestRegressor", "LinearRegression", "DecisionTreeRegressor"]
    assert result["validation"]["status"] == "valid"


def test_tune_decision_tree_grid(tmp_path: Path) -> None:
    """Verify that tune_single_model runs GridSearchCV strictly on training data."""
    df = load_data(sep=";")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    result = tune_single_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_family="decision_tree",
        search_strategy="grid",
        cv_folds=3,
        random_state=42,
    )

    assert result["model_family"] == "decision_tree"
    assert result["search_strategy"] == "GridSearchCV"
    assert "best_hyperparameters" in result
    assert "performance_comparison" in result
    assert "cv_mae" in result["performance_comparison"]["untuned"]
    assert "cv_mae" in result["performance_comparison"]["tuned"]
    assert "test_mae" in result["performance_comparison"]["tuned"]
    # Verify reproducible seed and documented search space
    assert result["random_state"] == 42
    assert "regressor__max_depth" in result["search_space"]


def test_tune_random_forest_randomized(tmp_path: Path) -> None:
    """Verify that tune_single_model runs RandomizedSearchCV strictly on training data."""
    df = load_data(sep=";")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    result = tune_single_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_family="random_forest",
        search_strategy="random",
        n_iter=3,
        cv_folds=3,
        random_state=42,
    )

    assert result["model_family"] == "random_forest"
    assert result["search_strategy"] == "RandomizedSearchCV"
    assert "regressor__n_estimators" in result["search_space"]
    assert result["performance_comparison"]["tuned"]["test_mae"] > 0.0


def test_tune_hyperparameters_both(tmp_path: Path) -> None:
    """Verify tune_hyperparameters tunes both models and outputs summary JSON."""
    df = load_data(sep=";")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    result = tune_hyperparameters(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_family="both",
        search_strategy="grid",
        cv_folds=3,
        random_state=42,
        reports_dir=tmp_path,
    )

    assert (tmp_path / "hyperparameter_tuning.json").exists()
    assert "models" in result
    assert "decision_tree" in result["models"]
    assert "random_forest" in result["models"]
    assert result["overall_best_model"] in ["random_forest", "decision_tree"]


def test_evaluation_report_content(tmp_path: Path) -> None:
    """Verify evaluation_report.md contains metric explanations and non-causal disclaimer."""
    reports_dir = tmp_path / "report_test"
    evaluate_model(reports_dir=reports_dir)

    report_md = reports_dir / "evaluation_report.md"
    assert report_md.exists()

    content = report_md.read_text(encoding="utf-8")
    assert "Mean Absolute Error (MAE)" in content
    assert "Root Mean Squared Error (RMSE)" in content
    assert "Mean Squared Error (MSE)" in content
    assert "Coefficient of Determination" in content
    assert "NOT causal effects" in content
    assert "0.4220" in content
    assert "0.5483" in content
    assert "0.5399" in content


