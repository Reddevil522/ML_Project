"""Unit tests for the preprocessing module."""

from pathlib import Path
import pytest
import pandas as pd

from src.preprocessing import (
    FEATURE_NAMES,
    TARGET_COLUMN,
    create_preprocessor,
    get_default_data_path,
    load_data,
    split_data,
    split_features_target,
)


def test_feature_definitions() -> None:
    """Verify that exactly 11 input features and 1 target are specified."""
    assert len(FEATURE_NAMES) == 11
    assert TARGET_COLUMN == "quality"
    assert "alcohol" in FEATURE_NAMES
    assert "volatile acidity" in FEATURE_NAMES
    assert "pH" in FEATURE_NAMES


def test_load_data_success() -> None:
    """Verify load_data parses semicolon-separated UCI CSV correctly."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df.columns) == 12  # 11 features + 1 target
    for feature in FEATURE_NAMES:
        assert feature in df.columns
    assert TARGET_COLUMN in df.columns
    assert df.isnull().sum().sum() == 0


def test_load_data_file_not_found(tmp_path: Path) -> None:
    """Verify load_data raises FileNotFoundError with descriptive message."""
    missing_file = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_data(missing_file)
    assert "Dataset not found" in str(exc_info.value)


def test_load_data_missing_columns(tmp_path: Path) -> None:
    """Verify load_data raises ValueError if required features are missing."""
    invalid_csv = tmp_path / "invalid.csv"
    # Create CSV missing several features
    invalid_df = pd.DataFrame({"fixed acidity": [7.4], "alcohol": [9.4]})
    invalid_df.to_csv(invalid_csv, sep=";", index=False)

    with pytest.raises(ValueError) as exc_info:
        load_data(invalid_csv)
    assert "missing required columns" in str(exc_info.value)


def test_split_features_target() -> None:
    """Verify split_features_target correctly separates X and y."""
    df = load_data()
    X, y = split_features_target(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[1] == 11
    assert len(X) == len(y)
    assert y.name == TARGET_COLUMN


def test_split_data_shapes() -> None:
    """Verify split_data produces reproducible train and test splits."""
    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    total_samples = len(df)
    assert len(X_train) + len(X_test) == total_samples
    assert len(y_train) + len(y_test) == total_samples
    assert X_train.shape[1] == 11
    assert X_test.shape[1] == 11
    assert len(X_test) == round(total_samples * 0.2)


def test_create_preprocessor() -> None:
    """Verify standard scaler preprocessor creation."""
    preprocessor = create_preprocessor()
    assert hasattr(preprocessor, "fit")
    assert hasattr(preprocessor, "transform")


def test_load_data_wrong_delimiter(tmp_path: Path) -> None:
    """Verify load_data raises ValueError if wrong delimiter is specified."""
    comma_csv = tmp_path / "comma_delimited.csv"
    df = pd.DataFrame({
        "fixed acidity": [7.4],
        "volatile acidity": [0.7],
        "citric acid": [0.0],
        "residual sugar": [1.9],
        "chlorides": [0.076],
        "free sulfur dioxide": [11.0],
        "total sulfur dioxide": [34.0],
        "density": [0.9978],
        "pH": [3.51],
        "sulphates": [0.56],
        "alcohol": [9.4],
        "quality": [5],
    })
    # Save with comma delimiter
    df.to_csv(comma_csv, sep=",", index=False)

    # Attempting to load with default ';' delimiter should fail
    with pytest.raises(ValueError) as exc_info:
        load_data(comma_csv, sep=";")
    assert "Invalid CSV structure" in str(exc_info.value) or "missing required columns" in str(exc_info.value)


def test_validate_dataset() -> None:
    """Verify validate_dataset accurately profiles dataset dimensions, types, and stats."""
    from src.preprocessing import validate_dataset

    df = load_data()
    summary = validate_dataset(df)

    assert summary["row_count"] == 1599
    assert summary["column_count"] == 12
    assert summary["total_missing_values"] == 0
    assert summary["total_infinite_values"] == 0
    assert summary["duplicate_rows_count"] == 240
    assert "chemical_feature_summary" in summary
    assert "target_quality_distribution" in summary
    assert summary["target_quality_distribution"]["min_quality"] == 3
    assert summary["target_quality_distribution"]["max_quality"] == 8

    # Verify each feature summary has count, mean, std, min, median, max
    for feat in FEATURE_NAMES:
        feat_stats = summary["chemical_feature_summary"][feat]
        assert "count" in feat_stats
        assert "mean" in feat_stats
        assert "std" in feat_stats
        assert "min" in feat_stats
        assert "median" in feat_stats
        assert "max" in feat_stats
        assert feat_stats["count"] == 1599


def test_save_validation_reports(tmp_path: Path) -> None:
    """Verify save_validation_reports produces data_summary.json and preprocessing_decisions.json."""
    from src.preprocessing import save_validation_reports

    df = load_data()
    summary_path, decisions_path = save_validation_reports(df, reports_dir=tmp_path)

    assert summary_path.exists()
    assert decisions_path.exists()
    assert summary_path.name == "data_summary.json"
    assert decisions_path.name == "preprocessing_decisions.json"


def test_preprocessor_prevents_leakage() -> None:
    """Verify preprocessor fits only on training data and transforms test data cleanly."""
    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    preprocessor = create_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    assert X_train_trans.shape == (1279, 11)
    assert X_test_trans.shape == (320, 11)

