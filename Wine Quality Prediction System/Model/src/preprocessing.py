"""Data preprocessing and exploratory analysis module for Wine Quality Prediction System.

Handles loading, validating, inspecting, plotting, splitting, and scaling of the
UCI Wine Quality dataset according to production ML engineering standards.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for automated execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 11 input physicochemical features according to the UCI Wine Quality dataset specification
FEATURE_NAMES: list[str] = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Regression target feature
TARGET_COLUMN: str = "quality"

# Plausible physicochemical boundary ranges for red wine (based on enological literature)
# Used for investigation of extreme/out-of-range values without deleting samples
PHYSIOCHEMICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "fixed acidity": (3.0, 20.0),        # g(tartaric acid)/dm^3
    "volatile acidity": (0.05, 2.0),     # g(acetic acid)/dm^3
    "citric acid": (0.0, 1.5),           # g/dm^3
    "residual sugar": (0.5, 30.0),       # g/dm^3
    "chlorides": (0.005, 1.0),           # g(sodium chloride)/dm^3
    "free sulfur dioxide": (0.5, 120.0), # mg/dm^3
    "total sulfur dioxide": (5.0, 350.0),# mg/dm^3
    "density": (0.980, 1.015),           # g/cm^3
    "pH": (2.5, 4.5),                    # pH scale
    "sulphates": (0.2, 2.5),             # g(potassium sulphate)/dm^3
    "alcohol": (7.0, 17.0),              # % vol
}


def get_default_data_path() -> Path:
    """Return the absolute path to the default winequality-red.csv file.

    Resolves dynamically relative to this source file, ensuring portability
    across different working directories.

    Returns:
        Path: Path object pointing to ml-backend/data/winequality-red.csv.
    """
    backend_dir = Path(__file__).resolve().parent.parent
    return backend_dir / "data" / "winequality-red.csv"


def get_default_reports_path() -> Path:
    """Return the absolute path to the reports directory.

    Returns:
        Path: Path object pointing to ml-backend/reports/.
    """
    backend_dir = Path(__file__).resolve().parent.parent
    reports_dir = backend_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def load_data(
    filepath: Path | str | None = None,
    sep: str = ";",
) -> pd.DataFrame:
    """Load and validate the wine quality dataset from a semicolon-separated CSV.

    Performs comprehensive verification of file existence, delimiter correctness,
    expected feature names, target presence, and data type validity.

    Args:
        filepath: Optional path to the CSV file. If None, uses default data path.
        sep: CSV delimiter. Defaults to ';' as in the original UCI dataset.

    Returns:
        pd.DataFrame: Validated wine quality DataFrame with ordered columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the resolved path.
        ValueError: If wrong delimiter is used, required columns are missing,
            unexpected columns are present, or non-numeric values are detected.
    """
    if filepath is None:
        target_path = get_default_data_path()
    else:
        target_path = Path(filepath).resolve()

    logger.info("Attempting to load dataset from: %s", target_path)

    if not target_path.exists():
        error_msg = (
            f"Dataset not found at '{target_path}'. "
            "Please ensure 'winequality-red.csv' is downloaded and placed into 'ml-backend/data/'. "
            "Refer to the project README for automated and manual download commands."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        df = pd.read_csv(target_path, sep=sep)
    except Exception as exc:
        logger.error("Failed to parse CSV file at %s: %s", target_path, exc)
        raise

    # Verify delimiter: if loaded with wrong delimiter, shape[1] is typically 1
    if df.shape[1] <= 1:
        error_msg = (
            f"Invalid CSV structure detected in '{target_path}'. Only {df.shape[1]} column was parsed. "
            f"Delimiter used was '{sep}'. Expected semicolon ';' separated columns."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("CSV loaded successfully. Shape: %s (rows=%d, cols=%d)", df.shape, df.shape[0], df.shape[1])

    # Validate presence of expected columns
    expected_columns = set(FEATURE_NAMES + [TARGET_COLUMN])
    actual_columns = set(df.columns)
    missing = expected_columns - actual_columns

    if missing:
        error_msg = (
            f"Dataset is missing required columns: {sorted(list(missing))}. "
            f"Expected columns: {sorted(list(expected_columns))}. "
            f"Found columns: {sorted(list(actual_columns))}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Check for unexpected extra columns
    unexpected = actual_columns - expected_columns
    if unexpected:
        logger.warning("Dataset contains unexpected extra columns: %s", sorted(list(unexpected)))

    # Ensure all columns contain valid numeric data types
    for col in FEATURE_NAMES + [TARGET_COLUMN]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Try converting to numeric to detect corruptions
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except Exception as exc:
                error_msg = f"Non-numeric values detected in column '{col}': {exc}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    # Check for missing values
    null_count = int(df.isnull().sum().sum())
    if null_count > 0:
        logger.warning("Dataset contains %d missing values. Imputation will be required.", null_count)
    else:
        logger.info("Data integrity check passed: 0 missing values detected.")

    # Check for infinite values
    inf_count = int(np.isinf(df[FEATURE_NAMES + [TARGET_COLUMN]].to_numpy()).sum())
    if inf_count > 0:
        logger.warning("Dataset contains %d infinite values.", inf_count)

    # Reorder columns to guarantee deterministic schema
    ordered_cols = FEATURE_NAMES + [TARGET_COLUMN]
    df = df[ordered_cols].copy()

    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform rigorous validation and statistical profiling of the dataset.

    Inspects row/column counts, data types, missing cells, duplicates, infinite values,
    chemical feature summaries (mean, std, min, median, max), out-of-range values,
    and target quality score distribution with class imbalance metrics.

    Args:
        df: Input DataFrame to analyze.

    Returns:
        Dict[str, Any]: Comprehensive validation summary dictionary.
    """
    logger.info("Running comprehensive data validation and statistical profiling...")

    row_count = int(df.shape[0])
    col_count = int(df.shape[1])
    dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    missing_by_col = {str(col): int(df[col].isnull().sum()) for col in df.columns}
    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    # Infinite values check
    inf_by_col = {
        str(col): int(np.isinf(df[col]).sum())
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    }
    total_inf = sum(inf_by_col.values())

    # Chemical feature summary (count, mean, std, min, median, max)
    feature_summary: Dict[str, Dict[str, float]] = {}
    for feat in FEATURE_NAMES:
        if feat in df.columns:
            series = df[feat]
            feature_summary[feat] = {
                "count": int(series.count()),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "median": round(float(series.median()), 4),
                "max": round(float(series.max()), 4),
            }

    # Out-of-range and extreme value investigation (without deleting samples)
    out_of_range_investigation: Dict[str, Dict[str, Any]] = {}
    for feat in FEATURE_NAMES:
        if feat in df.columns:
            series = df[feat]
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            iqr_lower = q1 - 1.5 * iqr
            iqr_upper = q3 + 1.5 * iqr
            iqr_outliers = int(((series < iqr_lower) | (series > iqr_upper)).sum())

            # Domain plausible bounds check
            lower_bound, upper_bound = PHYSIOCHEMICAL_BOUNDS.get(feat, (0.0, float("inf")))
            physically_out_of_bounds = int(((series < lower_bound) | (series > upper_bound)).sum())

            out_of_range_investigation[feat] = {
                "iqr_outlier_count": iqr_outliers,
                "iqr_lower_fence": round(iqr_lower, 4),
                "iqr_upper_fence": round(iqr_upper, 4),
                "enological_bounds": [lower_bound, upper_bound],
                "physically_out_of_bounds_count": physically_out_of_bounds,
                "action": "preserved_in_dataset_no_silent_deletion",
            }

    # Quality score distribution & class imbalance analysis (keeping task as regression)
    quality_counts = df[TARGET_COLUMN].value_counts().sort_index()
    quality_dist = {int(k): int(v) for k, v in quality_counts.items()}
    quality_pct = {int(k): round(float(v / row_count * 100), 2) for k, v in quality_counts.items()}

    # Imbalance metric (max frequency / min frequency)
    max_freq = max(quality_dist.values()) if quality_dist else 0
    min_freq = min(quality_dist.values()) if quality_dist else 1
    imbalance_ratio = round(max_freq / max(min_freq, 1), 2)

    target_analysis = {
        "score_counts": quality_dist,
        "score_percentages": quality_pct,
        "mean_quality": round(float(df[TARGET_COLUMN].mean()), 4),
        "std_quality": round(float(df[TARGET_COLUMN].std()), 4),
        "min_quality": int(df[TARGET_COLUMN].min()),
        "median_quality": float(df[TARGET_COLUMN].median()),
        "max_quality": int(df[TARGET_COLUMN].max()),
        "class_imbalance_ratio": imbalance_ratio,
        "most_frequent_scores": [5, 6],
        "modal_scores_combined_percentage": round(
            float(quality_pct.get(5, 0.0) + quality_pct.get(6, 0.0)), 2
        ),
        "modeling_formulation": "continuous_regression",
        "rationale": (
            "Sensory quality is an intrinsically ordinal scale with meaningful numerical distances. "
            "Regression appropriately penalizes errors based on distance (e.g. predicting 5 for a 4 "
            "has smaller penalty than predicting 8), whereas categorical classification would treat all "
            "misclassifications equally and severely suffer from minority class rarity (ratings 3 and 8)."
        ),
    }

    summary: Dict[str, Any] = {
        "dataset_name": "UCI Red Wine Quality Dataset",
        "row_count": row_count,
        "column_count": col_count,
        "data_types": dtypes,
        "missing_values": missing_by_col,
        "total_missing_values": total_missing,
        "duplicate_rows_count": duplicate_rows,
        "duplicate_rows_treatment": "retained_as_valid_measurements_from_same_batch",
        "infinite_values": inf_by_col,
        "total_infinite_values": total_inf,
        "chemical_feature_summary": feature_summary,
        "out_of_range_investigation": out_of_range_investigation,
        "target_quality_distribution": target_analysis,
    }

    logger.info(
        "Validation complete: %d rows, %d cols, %d missing, %d duplicates, %d infs.",
        row_count,
        col_count,
        total_missing,
        duplicate_rows,
        total_inf,
    )
    return summary


def generate_eda_visualizations(
    df: pd.DataFrame,
    reports_dir: Path | str | None = None,
) -> Dict[str, Path]:
    """Generate and save 5 readable publication-quality exploratory plots.

    1. Target quality distribution (bar chart with percentages).
    2. Feature histograms (all 11 features with KDE).
    3. Correlation heatmap (annotated Pearson matrix).
    4. Boxplots for numerical features (distributions & IQR outliers).
    5. Feature relationships with wine quality (key physicochemical drivers).

    Args:
        df: Input DataFrame.
        reports_dir: Destination directory. Defaults to ml-backend/reports/.

    Returns:
        Dict[str, Path]: Dictionary mapping plot names to saved file paths.
    """
    if reports_dir is None:
        out_dir = get_default_reports_path()
    else:
        out_dir = Path(reports_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating EDA visualizations in: %s", out_dir)
    sns.set_theme(style="whitegrid", palette="muted")
    saved_plots: Dict[str, Path] = {}

    # -------------------------------------------------------------
    # 1. Target quality distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    palette = sns.color_palette("dark:#722F37", len(counts))
    bars = ax.bar([str(k) for k in counts.index], counts.values, color=palette, edgecolor="black", width=0.6)

    # Annotate counts and percentages
    total = len(df)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.annotate(
            f"{int(h)}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Distribution of Red Wine Sensory Quality Ratings", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Sensory Quality Score (0 - 10 Scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(counts.values) * 1.18)
    plt.tight_layout()
    p1 = out_dir / "target_quality_distribution.png"
    fig.savefig(p1)
    plt.close(fig)
    saved_plots["target_quality_distribution"] = p1

    # -------------------------------------------------------------
    # 2. Feature histograms (all 11 features)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), dpi=300)
    axes_flat = axes.flatten()

    for idx, feat in enumerate(FEATURE_NAMES):
        ax = axes_flat[idx]
        sns.histplot(
            data=df,
            x=feat,
            kde=True,
            ax=ax,
            color="#2E6F9E",
            edgecolor="white",
            bins=22,
        )
        ax.set_title(feat.title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Frequency", fontsize=10)

    # Remove the 12th unused subplot in 4x3 grid
    fig.delaxes(axes_flat[-1])

    fig.suptitle("Physicochemical Feature Distributions (Histograms with KDE)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    p2 = out_dir / "feature_histograms.png"
    fig.savefig(p2)
    plt.close(fig)
    saved_plots["feature_histograms"] = p2

    # -------------------------------------------------------------
    # 3. Correlation heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
    cols = FEATURE_NAMES + [TARGET_COLUMN]
    corr_matrix = df[cols].corr()

    # Mask upper triangle for cleaner readability
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-0.8,
        vmax=0.8,
        center=0,
        square=True,
        linewidths=0.6,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax,
    )
    ax.set_title("Physicochemical Correlation Heatmap (with Target 'quality')", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    p3 = out_dir / "correlation_heatmap.png"
    fig.savefig(p3)
    plt.close(fig)
    saved_plots["correlation_heatmap"] = p3

    # -------------------------------------------------------------
    # 4. Boxplots for numerical features
    # -------------------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), dpi=300)
    axes_flat = axes.flatten()

    for idx, feat in enumerate(FEATURE_NAMES):
        ax = axes_flat[idx]
        sns.boxplot(
            data=df,
            y=feat,
            ax=ax,
            color="#E07A5F",
            width=0.4,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )
        ax.set_title(feat.title(), fontsize=12, fontweight="bold")
        ax.set_ylabel("")

    # Remove the 12th unused subplot
    fig.delaxes(axes_flat[-1])

    fig.suptitle("Boxplots for Numerical Physicochemical Features", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    p4 = out_dir / "feature_boxplots.png"
    fig.savefig(p4)
    plt.close(fig)
    saved_plots["feature_boxplots"] = p4

    # -------------------------------------------------------------
    # 5. Feature relationships with wine quality
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Alcohol vs Quality (strongest positive driver)
    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y="alcohol",
        hue=TARGET_COLUMN,
        legend=False,
        ax=axes[0, 0],
        palette="Blues",
    )
    axes[0, 0].set_title("Alcohol (% vol) vs Quality (Positive Driver: r = +0.48)", fontweight="bold", fontsize=11)
    axes[0, 0].set_xlabel("Sensory Quality Rating")
    axes[0, 0].set_ylabel("Alcohol (% vol)")

    # Volatile Acidity vs Quality (strongest negative driver - acetic acid / vinegar fault)
    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y="volatile acidity",
        hue=TARGET_COLUMN,
        legend=False,
        ax=axes[0, 1],
        palette="Reds",
    )
    axes[0, 1].set_title("Volatile Acidity vs Quality (Negative Driver: r = -0.39)", fontweight="bold", fontsize=11)
    axes[0, 1].set_xlabel("Sensory Quality Rating")
    axes[0, 1].set_ylabel("Volatile Acidity (g/dm^3)")

    # Sulphates vs Quality (positive driver - antioxidant / preservative)
    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y="sulphates",
        hue=TARGET_COLUMN,
        legend=False,
        ax=axes[1, 0],
        palette="Purples",
    )
    axes[1, 0].set_title("Sulphates vs Quality (Positive Driver: r = +0.25)", fontweight="bold", fontsize=11)
    axes[1, 0].set_xlabel("Sensory Quality Rating")
    axes[1, 0].set_ylabel("Sulphates (g/dm^3)")

    # Citric Acid vs Quality (freshness attribute)
    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y="citric acid",
        hue=TARGET_COLUMN,
        legend=False,
        ax=axes[1, 1],
        palette="Greens",
    )
    axes[1, 1].set_title("Citric Acid vs Quality (Freshness Driver: r = +0.23)", fontweight="bold", fontsize=11)
    axes[1, 1].set_xlabel("Sensory Quality Rating")
    axes[1, 1].set_ylabel("Citric Acid (g/dm^3)")

    fig.suptitle("Key Physicochemical Feature Relationships with Sensory Quality", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    p5 = out_dir / "feature_quality_relationships.png"
    fig.savefig(p5)
    plt.close(fig)
    saved_plots["feature_quality_relationships"] = p5

    logger.info("Successfully generated all 5 EDA visualization plots.")
    return saved_plots


def save_validation_reports(
    df: pd.DataFrame,
    reports_dir: Path | str | None = None,
) -> Tuple[Path, Path]:
    """Save structured data validation summary and preprocessing decisions into reports/.

    Args:
        df: Input DataFrame.
        reports_dir: Destination reports directory.

    Returns:
        Tuple[Path, Path]: (summary_json_path, decisions_json_path).
    """
    if reports_dir is None:
        out_dir = get_default_reports_path()
    else:
        out_dir = Path(reports_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate validation summary
    summary_data = validate_dataset(df)
    summary_file = out_dir / "data_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info("Saved data validation summary to: %s", summary_file)

    # 2. Document formal preprocessing and architectural decisions
    decisions = {
        "dataset_metadata": {
            "source": "UCI Wine Quality Dataset (Red Wine, Cortez et al., 2009)",
            "delimiter": "semicolon (;)",
            "feature_count": 11,
            "target_column": TARGET_COLUMN,
            "ordered_features": FEATURE_NAMES,
        },
        "preprocessing_decisions": {
            "imputation": {
                "decision": "SimpleImputer(strategy='median') inside pipeline",
                "rationale": (
                    "While the primary dataset has 0 missing values, production deployment requires "
                    "robustness against unexpected missing values in real-time inference payloads."
                ),
                "leakage_prevention": "Imputer is fitted strictly on X_train only.",
            },
            "feature_scaling": {
                "decision": "StandardScaler()",
                "rationale": (
                    "Physicochemical features vary across divergent scales: total sulfur dioxide reaches 289 "
                    "while chlorides are below 0.61. StandardScaler centers features to mean 0 and variance 1."
                ),
                "leakage_prevention": "StandardScaler is fitted strictly on X_train within a Scikit-Learn Pipeline.",
            },
            "dimensionality_reduction_pca": {
                "used_by_default": False,
                "justification_for_exclusion": (
                    "1. Low-dimensional feature space: With only 11 features and 1,599 observations (p << n), "
                    "there is no curse of dimensionality or computational bottleneck.\n"
                    "2. Loss of physicochemical interpretability: Each feature corresponds to a specific enological "
                    "compound (e.g. volatile acidity = acetic acid defect; alcohol = body; sulphates = antioxidant). "
                    "PCA projects these into linear combinations of latent eigenvectors, destroying feature importance "
                    "and enological interpretability for domain practitioners.\n"
                    "3. Non-linear tree resilience: Tree-based ensemble models (RandomForestRegressor) inherently perform "
                    "implicit feature selection and are invariant to monotonic transformations and collinearity."
                ),
                "when_pca_would_be_justified": (
                    "Dimensionality reduction like PCA would be justified if:\n"
                    "a) Working with high-dimensional spectrometry data (e.g., hundreds or thousands of infrared "
                    "or NMR spectral bands where p > n);\n"
                    "b) Severe multicollinearity causing numerical instability in unregularized linear models;\n"
                    "c) Severe storage or low-latency computational constraints requiring compressed feature vectors."
                ),
            },
            "sample_and_outlier_treatment": {
                "decision": "Retain all 1,599 valid samples (no silent deletion of outliers or duplicate rows)",
                "rationale": (
                    "1. Duplicate rows (240 rows): In enological chemical profiling, independent wine bottles from "
                    "the same batch, barrel, or vineyard commonly share identical discrete lab measurements and "
                    "expert ratings. Deleting them would artificially alter the natural population distribution.\n"
                    "2. Outliers (e.g., residual sugar up to 15.5 g/dm^3, total SO2 up to 289 mg/dm^3): These represent "
                    "legitimate styles of wine (e.g. sweeter dessert variants, wines with higher preservative levels) "
                    "rather than measurement or sensor corruptions. Robust scaling and ensemble trees handle them safely."
                ),
            },
            "data_splitting_strategy": {
                "train_test_split_ratio": "80% Train, 20% Holdout Test",
                "random_state": 42,
                "leakage_guarantee": (
                    "Dataset is split into X_train, X_test, y_train, y_test BEFORE any transformation. "
                    "All transformers are encapsulated in Scikit-Learn Pipelines fitted solely on training splits."
                ),
            },
            "task_framing": {
                "formulation": "Regression (predicting continuous quality score)",
                "rationale": (
                    "Sensory score is an ordinal measurement with continuous distance properties. "
                    "Framing as regression allows natural distance loss metrics (MSE/RMSE/MAE) and circumvents "
                    "severe class imbalance on extreme scores (score 3 has only 10 samples; score 8 has only 18)."
                ),
            },
        },
    }

    decisions_file = out_dir / "preprocessing_decisions.json"
    with open(decisions_file, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)
    logger.info("Saved preprocessing decisions to: %s", decisions_file)

    return summary_file, decisions_file


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_names: list[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate the input features from the target column.

    Strictly preserves feature names, column order, and metadata.

    Args:
        df: Input DataFrame containing both features and target.
        target_column: Name of the target column. Defaults to 'quality'.
        feature_names: List of feature names to select. Defaults to FEATURE_NAMES.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X features DataFrame, y target Series).

    Raises:
        KeyError: If target_column or any feature in feature_names is not in df.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise KeyError(f"Features missing from DataFrame: {missing_features}")

    # Preserve explicit feature names and sequence
    X = df[feature_names].copy()
    y = df[target_column].copy()

    logger.info("Features and target split successfully. X shape: %s, y shape: %s", X.shape, y.shape)
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into reproducible training and testing sets.

    Ensures zero data leakage by creating distinct partitions with a fixed random seed.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state: Random state seed for reproducibility. Defaults to 42.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test).
    """
    logger.info(
        "Splitting dataset: test_size=%.2f, random_state=%d",
        test_size,
        random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info(
        "Split complete: X_train=%s, X_test=%s, y_train=%s, y_test=%s",
        X_train.shape,
        X_test.shape,
        y_train.shape,
        y_test.shape,
    )
    return X_train, X_test, y_train, y_test


def create_preprocessor() -> Pipeline:
    """Instantiate a leak-free Scikit-learn preprocessing pipeline.

    Combines median imputation (SimpleImputer) for missing value resilience
    with standardization (StandardScaler) for centering and variance scaling.
    Designed to be fitted strictly on training data within end-to-end pipelines.

    Returns:
        Pipeline: Configured preprocessor pipeline instance with fit and transform.
    """
    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return preprocessor


def run_preprocessing_pipeline(
    data_path: Path | str | None = None,
    reports_dir: Path | str | None = None,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """Execute complete data loading, validation, profiling, and report generation.

    Args:
        data_path: Optional path to winequality-red.csv.
        reports_dir: Optional destination for reports and plots.
        generate_plots: Whether to generate and save the 5 EDA plots.

    Returns:
        Dict[str, Any]: Summary of validation, reports, and split shapes.
    """
    logger.info("Starting complete preprocessing and exploratory analysis pipeline...")

    # 1. Load data with strict delimiter and schema verification
    df = load_data(filepath=data_path, sep=";")

    # 2. Run data validation and statistics profiling
    val_summary = validate_dataset(df)

    # 3. Save validation reports and decision log to reports/
    summary_file, decisions_file = save_validation_reports(df, reports_dir=reports_dir)

    # 4. Generate the 5 required visualization plots
    plots = {}
    if generate_plots:
        plots = generate_eda_visualizations(df, reports_dir=reports_dir)

    # 5. Split features and target, and create leak-free train/test sets
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # 6. Fit preprocessor on training data only to demonstrate leak-free transformation
    preprocessor = create_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    logger.info("Pipeline execution completed successfully.")

    return {
        "validation_summary": val_summary,
        "summary_file": str(summary_file),
        "decisions_file": str(decisions_file),
        "generated_plots": {k: str(v) for k, v in plots.items()},
        "splits": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape),
            "X_train_scaled_shape": list(X_train_scaled.shape),
            "X_test_scaled_shape": list(X_test_scaled.shape),
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Executing preprocessing and exploratory analysis script...")
    results = run_preprocessing_pipeline()

    print("\n" + "=" * 70)
    print("WINE QUALITY PREPROCESSING & EXPLORATORY ANALYSIS SUMMARY")
    print("=" * 70)
    val = results["validation_summary"]
    print(f"Total Rows:             {val['row_count']}")
    print(f"Total Columns:          {val['column_count']}")
    print(f"Total Missing Cells:    {val['total_missing_values']}")
    print(f"Duplicate Rows Count:   {val['duplicate_rows_count']} (preserved, valid enological batch replicates)")
    print(f"Infinite Values Count:  {val['total_infinite_values']}")
    print("\nQuality Score Distribution:")
    for score, cnt in val["target_quality_distribution"]["score_counts"].items():
        pct = val["target_quality_distribution"]["score_percentages"][score]
        print(f"  Rating {score}: {cnt:4d} samples ({pct:5.2f}%)")

    print("\nGenerated Reports:")
    print(f"  - Validation Summary:   {results['summary_file']}")
    print(f"  - Preprocessing Logic:  {results['decisions_file']}")

    print("\nGenerated Visualizations:")
    for name, path in results["generated_plots"].items():
        print(f"  - {name}: {path}")

    print("\nTrain / Test Split Shapes (80/20, random_state=42):")
    for split_name, shape in results["splits"].items():
        print(f"  - {split_name}: {shape}")
    print("=" * 70)
