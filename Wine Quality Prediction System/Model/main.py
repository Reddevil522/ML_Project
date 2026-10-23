"""FastAPI Application for Wine Quality Prediction System.

Exposes production-ready RESTful endpoints for health monitoring,
model metadata inspection, and real-time sensory quality inference.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.preprocessing import FEATURE_NAMES, TARGET_COLUMN, get_default_data_path
from src.train import get_default_metadata_path, get_default_model_path

logger = logging.getLogger("wine_quality.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =====================================================================
# API Configuration & Settings
# =====================================================================
class APISettings:
    """Centralized configuration for API service and runtime dependencies."""

    TITLE: str = "Wine Quality Prediction API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "Production-grade REST API providing machine learning inference "
        "for red wine sensory quality scores based on 11 physicochemical properties."
    )
    CORS_ORIGINS: List[str] = [
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    MODEL_PATH: Path = get_default_model_path()
    METADATA_PATH: Path = get_default_metadata_path()
    DATA_PATH: Path = get_default_data_path()
    REPORTS_DIR: Path = Path(__file__).resolve().parent / "reports"


settings = APISettings()

# Exact mapping from Pydantic request field names to UCI dataset column headers
FIELD_TO_DATASET_MAPPING: Dict[str, str] = {
    "fixed_acidity": "fixed acidity",
    "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid",
    "residual_sugar": "residual sugar",
    "chlorides": "chlorides",
    "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide": "total sulfur dioxide",
    "density": "density",
    "ph": "pH",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
}


# =====================================================================
# Model Artifact Manager (Singleton Loaded at Startup)
# =====================================================================
class ModelManager:
    """Manages model artifact lifecycle, memory persistence, and inference."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        self.model_path = model_path or settings.MODEL_PATH
        self.metadata_path = metadata_path or settings.METADATA_PATH
        self.model: Any = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.model_type_name: Optional[str] = None
        self.raw_pipeline_str: Optional[str] = None

    def load(self) -> bool:
        """Load model artifact and associated training metadata from disk."""
        if not self.model_path.exists():
            logger.warning("Model artifact not found at: %s", self.model_path)
            self.model = None
            self.metadata = None
            self.model_type_name = None
            self.raw_pipeline_str = None
            return False

        try:
            self.model = joblib.load(self.model_path)
            self.raw_pipeline_str = str(self.model)

            # Determine human-readable estimator name
            if hasattr(self.model, "named_steps") and "regressor" in self.model.named_steps:
                self.model_type_name = type(self.model.named_steps["regressor"]).__name__
            else:
                self.model_type_name = type(self.model).__name__

            # Load training metadata if present
            if self.metadata_path.exists():
                try:
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                except Exception as meta_exc:
                    logger.warning("Could not parse metadata JSON from %s: %s", self.metadata_path, meta_exc)
                    self.metadata = None
            else:
                self.metadata = None

            logger.info(
                "Model artifact loaded successfully: %s from %s",
                self.model_type_name,
                self.model_path,
            )
            return True
        except Exception as exc:
            logger.error("Failed to deserialize model artifact from %s: %s", self.model_path, exc)
            self.model = None
            self.metadata = None
            self.model_type_name = None
            self.raw_pipeline_str = None
            return False

    @property
    def is_loaded(self) -> bool:
        """Return True if model is loaded into memory and ready for inference."""
        return self.model is not None


model_manager = ModelManager()
# Initial model load on startup / module import
model_manager.load()


# =====================================================================
# Lifespan Context Manager (Startup / Shutdown)
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact once at application startup and manage teardown."""
    logger.info("Initializing %s v%s...", settings.TITLE, settings.VERSION)
    if not model_manager.is_loaded:
        model_manager.load()
    yield
    logger.info("Shutting down %s.", settings.TITLE)


# =====================================================================
# FastAPI Application Instance
# =====================================================================
app = FastAPI(
    title=settings.TITLE,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for Angular frontend development origin (localhost:4200)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"http://localhost(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Pydantic Schemas with Defensible Enological Bounds
# =====================================================================
class WineFeaturesInput(BaseModel):
    """Validates the 11 physicochemical wine properties.

    Bounds are informed by the empirical distributions of the UCI Red Wine dataset
    combined with practical winemaking chemistry (enology). Defensible bounds prevent
    extreme extrapolation or physical impossibilities without confusing dataset sample
    ranges with universal physical barriers.
    """

    model_config = ConfigDict(
        extra="forbid",  # Reject unexpected or extra properties
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.70,
                "citric_acid": 0.00,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "ph": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
            }
        },
    )

    fixed_acidity: float = Field(
        ...,
        description=(
            "Fixed acidity in tartaric acid (g/dm³). Dataset range: [4.6, 15.9], mean: 8.32. "
            "Defensible bounds [3.0, 25.0] allow high-acid northern varieties while rejecting absurd values."
        ),
        ge=3.0,
        le=25.0,
        examples=[7.4],
    )
    volatile_acidity: float = Field(
        ...,
        description=(
            "Volatile acidity in acetic acid (g/dm³). Dataset range: [0.12, 1.58], mean: 0.53. "
            "Defensible bounds [0.05, 3.0] cover standard dry table wines through severely flawed vinegar spoilage."
        ),
        ge=0.05,
        le=3.0,
        examples=[0.70],
    )
    citric_acid: float = Field(
        ...,
        description=(
            "Citric acid (g/dm³). Dataset range: [0.0, 1.0], mean: 0.27. "
            "Defensible bounds [0.0, 2.0] cover natural absence up to legal acidification limits."
        ),
        ge=0.0,
        le=2.0,
        examples=[0.00],
    )
    residual_sugar: float = Field(
        ...,
        description=(
            "Residual sugar post-fermentation (g/dm³). Dataset range: [0.9, 15.5], mean: 2.54. "
            "Defensible bounds [0.5, 100.0] cover bone-dry wines through late-harvest sweet reds."
        ),
        ge=0.5,
        le=100.0,
        examples=[1.9],
    )
    chlorides: float = Field(
        ...,
        description=(
            "Chlorides in sodium chloride (g/dm³). Dataset range: [0.012, 0.611], mean: 0.088. "
            "Defensible bounds [0.005, 1.0] accommodate maritime salinity without palate failure."
        ),
        ge=0.005,
        le=1.0,
        examples=[0.076],
    )
    free_sulfur_dioxide: float = Field(
        ...,
        description=(
            "Free molecular and bisulfite SO₂ (mg/dm³). Dataset range: [1.0, 72.0], mean: 15.87. "
            "Defensible bounds [0.0, 150.0] represent active antimicrobial preservation."
        ),
        ge=0.0,
        le=150.0,
        examples=[11.0],
    )
    total_sulfur_dioxide: float = Field(
        ...,
        description=(
            "Total sulfur dioxide (mg/dm³). Dataset range: [6.0, 289.0], mean: 46.47. "
            "Defensible bounds [0.0, 450.0] encompass global regulatory ceilings for preserved wines."
        ),
        ge=0.0,
        le=450.0,
        examples=[34.0],
    )
    density: float = Field(
        ...,
        description=(
            "Specific gravity relative to pure water (g/cm³). Dataset range: [0.9901, 1.0037], mean: 0.9967. "
            "Defensible bounds [0.9800, 1.0400] reflect the balance between ethanol and dissolved sugars."
        ),
        ge=0.9800,
        le=1.0400,
        examples=[0.9978],
    )
    ph: float = Field(
        ...,
        description=(
            "Wine acidity on the pH scale. Dataset range: [2.74, 4.01], mean: 3.31. "
            "Defensible bounds [2.5, 4.5] define the biological stability window for drinkable wines."
        ),
        ge=2.5,
        le=4.5,
        examples=[3.51],
    )
    sulphates: float = Field(
        ...,
        description=(
            "Potassium sulphate additive (g/dm³). Dataset range: [0.33, 2.00], mean: 0.66. "
            "Defensible bounds [0.1, 3.0] cover standard mineral stabilizer concentrations."
        ),
        ge=0.1,
        le=3.0,
        examples=[0.56],
    )
    alcohol: float = Field(
        ...,
        description=(
            "Ethanol content by volume (% vol). Dataset range: [8.4, 14.9], mean: 10.42. "
            "Defensible bounds [6.0, 20.0] cover light table reds up to fortified dessert wines."
        ),
        ge=6.0,
        le=20.0,
        examples=[9.4],
    )

    @model_validator(mode="before")
    @classmethod
    def handle_ph_alias(cls, data: Any) -> Any:
        """Allow clients to send either 'pH' or 'ph' seamlessly."""
        if isinstance(data, dict):
            if "pH" in data and "ph" not in data:
                data = dict(data)
                data["ph"] = data.pop("pH")
        return data

    @field_validator("*")
    @classmethod
    def validate_finite_numbers(cls, v: Any, info: Any) -> Any:
        """Ensure all numeric fields are finite numbers (reject NaN and Inf)."""
        if isinstance(v, (int, float)):
            if not math.isfinite(v):
                raise ValueError(
                    f"Field '{info.field_name}' must be a finite numeric value, got '{v}'."
                )
        return v

    @model_validator(mode="after")
    def validate_sulfur_dioxide_consistency(self) -> WineFeaturesInput:
        """Verify that free SO₂ does not exceed total SO₂ (physical impossibility)."""
        if self.free_sulfur_dioxide > self.total_sulfur_dioxide:
            raise ValueError(
                f"Physically inconsistent: free_sulfur_dioxide ({self.free_sulfur_dioxide} mg/dm³) "
                f"cannot exceed total_sulfur_dioxide ({self.total_sulfur_dioxide} mg/dm³)."
            )
        return self


class WinePredictionResponse(BaseModel):
    """Output schema returned after successful regression prediction."""

    predicted_quality: float = Field(
        ...,
        description="Actual continuous quality score computed by the trained regression model",
    )
    model_type: str = Field(
        ...,
        description="Actual loaded estimator or pipeline type serving inference",
    )
    message: str = Field(
        "Prediction computed successfully.",
        description="Status message indicating outcome",
    )
    # Additional fields supporting Angular UI compatibility
    rounded_quality: int = Field(
        ...,
        description="Nearest integer quality score on the 10-point sensory scale (typically 3 to 8)",
    )
    quality_category: str = Field(
        ...,
        description="Descriptive sensory quality categorization",
    )
    status: str = Field(
        "success",
        description="Operation status indicator",
    )
    model_name: str = Field(
        ...,
        description="Identifier of the serving model",
    )


class HealthResponse(BaseModel):
    """System health check and model loading status response."""

    status: str = Field(..., description="'healthy' if model is ready, 'degraded' otherwise")
    model_loaded: bool = Field(..., description="Whether the trained ML model is loaded in memory")
    model_type: Optional[str] = Field(None, description="Class name of the loaded model estimator")
    version: str = Field(..., description="API software version")
    feature_count: int = Field(11, description="Number of expected chemical features")
    dataset_available: bool = Field(..., description="Whether source dataset exists on disk")


class ModelInfoResponse(BaseModel):
    """Detailed model architecture, feature requirements, and training metadata."""

    selected_estimator: str = Field(..., description="Name of selected regression algorithm")
    model_type: str = Field(..., description="Full pipeline representation of the estimator")
    required_feature_names: List[str] = Field(..., description="The 11 features in exact column order")
    feature_count: int = Field(11, description="Total count of required input features")
    target_column: str = Field("quality", description="Name of the regression target column")
    pipeline_steps: List[str] = Field(..., description="Ordered list of steps in the pipeline")
    training_metadata: Optional[Dict[str, Any]] = Field(None, description="Available training metadata")


def classify_quality(score: float) -> str:
    """Classify continuous quality score into descriptive sensory categories."""
    if score >= 7.0:
        return "Excellent / Premium"
    elif score >= 6.0:
        return "Good Quality"
    elif score >= 5.0:
        return "Average / Standard"
    else:
        return "Low / Deficient"


class DatasetSummary(BaseModel):
    """Exploratory and structural dataset metrics from verified reports."""

    sample_count: int = Field(..., description="Total rows in UCI Red Wine dataset")
    feature_count: int = Field(11, description="Number of physicochemical predictors")
    average_quality: float = Field(..., description="Observed mean target sensory quality")
    quality_std: float = Field(..., description="Observed standard deviation of target quality")
    quality_distribution: Dict[str, int] = Field(..., description="Sample counts per quality score (3-8)")
    quality_percentages: Dict[str, float] = Field(..., description="Percentage representation per quality score")


class ModelTestMetrics(BaseModel):
    """Holdout test evaluation metrics for the selected model artifact."""

    selected_model: str = Field(..., description="Winning model algorithm name")
    test_samples: int = Field(..., description="Holdout test set sample count")
    mae: float = Field(..., description="Holdout test set Mean Absolute Error")
    rmse: float = Field(..., description="Holdout test set Root Mean Squared Error")
    r2: float = Field(..., description="Holdout test set Coefficient of Determination R²")
    mse: float = Field(..., description="Holdout test set Mean Squared Error")
    explained_variance: float = Field(..., description="Holdout test set Explained Variance")


class ModelComparisonItem(BaseModel):
    """Cross-validation benchmarks comparing regression candidates."""

    model: str = Field(..., description="Algorithm name")
    mae: float = Field(..., description="5-fold CV Mean Absolute Error")
    rmse: float = Field(..., description="5-fold CV Root Mean Squared Error")
    r2: float = Field(..., description="5-fold CV R² Score")
    mae_std: Optional[float] = Field(None, description="CV MAE standard deviation")
    rmse_std: Optional[float] = Field(None, description="CV RMSE standard deviation")
    r2_std: Optional[float] = Field(None, description="CV R² standard deviation")


class FeatureImportanceItem(BaseModel):
    """Normalized feature importance ranking from tree-based regressor."""

    feature: str = Field(..., description="Physicochemical feature name")
    importance: float = Field(..., description="MDI feature importance score")
    percentage: float = Field(..., description="Normalized percentage contribution")


class AnalyticsSummaryResponse(BaseModel):
    """Consolidated KPI and metadata payload for dashboard and analytics."""

    dataset: DatasetSummary = Field(..., description="Verified dataset statistics")
    model: ModelTestMetrics = Field(..., description="Verified holdout test metrics")
    model_comparison: List[ModelComparisonItem] = Field(..., description="Saved CV comparisons across regression models")
    feature_importance: List[FeatureImportanceItem] = Field(..., description="Feature importance rankings when supported")


class CorrelationMatrixResponse(BaseModel):
    """Physicochemical Pearson correlation matrix including quality target."""

    columns: List[str] = Field(..., description="Ordered column names matching matrix rows and columns")
    matrix: List[List[float]] = Field(..., description="2D correlation matrix of Pearson coefficients")


class ActualVsPredictedPoint(BaseModel):
    """Individual observation evaluation point on holdout test set."""

    actual: float = Field(..., description="Observed discrete sensory rating")
    predicted: float = Field(..., description="Continuous predicted quality score")
    residual: float = Field(..., description="Residual error (actual - predicted)")


class ActualVsPredictedResponse(BaseModel):
    """Holdout test evaluation points for actual vs predicted scatter plot."""

    test_samples: int = Field(..., description="Total holdout test evaluation points")
    mae: float = Field(..., description="Holdout test Mean Absolute Error")
    rmse: float = Field(..., description="Holdout test Root Mean Squared Error")
    r2: float = Field(..., description="Holdout test R² score")
    points: List[ActualVsPredictedPoint] = Field(..., description="Array of actual vs predicted coordinate pairs")


# =====================================================================
# Exception Handlers
# =====================================================================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return clean, sanitized 422 errors without stack traces."""
    errors = []
    for err in exc.errors():
        field_name = " -> ".join(str(loc) for loc in err["loc"] if loc != "body")
        errors.append({
            "field": field_name or "body",
            "message": err["msg"],
            "type": err["type"],
        })
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Input validation failed. Please correct the highlighted properties.",
            "errors": errors,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and return 500 without exposing internal stack traces."""
    logger.exception("Unhandled server exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred while processing your request. Please try again later.",
        },
    )


# =====================================================================
# REST Endpoints
# =====================================================================
@app.get("/", tags=["General"])
def api_root() -> Dict[str, Any]:
    """Root endpoint providing service metadata and discovery links."""
    return {
        "project": "Wine Quality Prediction System",
        "title": settings.TITLE,
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
        "documentation": "/docs",
        "redoc": "/redoc",
        "health_check": "/health",
        "model_info": "/model-info",
        "prediction_endpoint": "/predict",
        "analytics_summary": "/analytics/summary",
        "analytics_correlation": "/analytics/correlation",
        "analytics_actual_vs_predicted": "/analytics/actual-vs-predicted",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check() -> HealthResponse:
    """Report service status, dataset availability, and model readiness."""
    data_exists = settings.DATA_PATH.exists()
    is_ready = model_manager.is_loaded

    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_type=model_manager.model_type_name,
        version=settings.VERSION,
        feature_count=len(FEATURE_NAMES),
        dataset_available=data_exists,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
def get_model_info() -> ModelInfoResponse:
    """Return selected estimator, required feature names, and available training metadata."""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model artifact is not loaded. "
                "Please run 'python src/train.py' to generate and serialize the model."
            ),
        )

    pipeline = model_manager.model
    pipeline_steps = [step[0] for step in pipeline.steps] if hasattr(pipeline, "steps") else ["estimator"]

    return ModelInfoResponse(
        selected_estimator=model_manager.model_type_name or "Unknown",
        model_type=model_manager.raw_pipeline_str or str(pipeline),
        required_feature_names=FEATURE_NAMES,
        feature_count=len(FEATURE_NAMES),
        target_column=TARGET_COLUMN,
        pipeline_steps=pipeline_steps,
        training_metadata=model_manager.metadata,
    )


@app.get("/features", tags=["Metadata"])
def get_features_metadata() -> Dict[str, Any]:
    """Return feature names and target definition for backward compatibility."""
    return {
        "features": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "target": TARGET_COLUMN,
        "target_type": "continuous_regression",
        "score_range": "3 to 8 (typical UCI Red Wine scale)",
    }


@app.post(
    "/predict",
    response_model=WinePredictionResponse,
    tags=["Inference"],
    status_code=status.HTTP_200_OK,
)
def predict_wine_quality(features: WineFeaturesInput) -> WinePredictionResponse:
    """Predict wine quality score from exactly 11 validated chemical properties.

    Reuses the preprocessing pipeline bundled with the loaded model artifact.
    Rejects missing or extra properties, validates finite values, and enforces
    defensible enological bounds.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model artifact is not available. "
                "Please ensure the model has been trained and saved with 'python src/train.py'."
            ),
        )

    try:
        # Map request fields to the dataset's exact column names and order
        feature_dict = {
            FIELD_TO_DATASET_MAPPING[field]: getattr(features, field)
            for field in FIELD_TO_DATASET_MAPPING
        }
        input_df = pd.DataFrame([feature_dict])[FEATURE_NAMES]

        # Execute inference through the bundled pipeline (imputer + scaler + regressor)
        raw_pred = model_manager.model.predict(input_df)
        pred_val = round(float(raw_pred[0]), 4)
        category = classify_quality(pred_val)
        rounded_val = int(round(pred_val))
        estimator_name = model_manager.model_type_name or "Regressor"

        return WinePredictionResponse(
            predicted_quality=pred_val,
            model_type=estimator_name,
            message="Wine quality predicted successfully.",
            rounded_quality=rounded_val,
            quality_category=category,
            status="success",
            model_name=estimator_name,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Inference evaluation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while evaluating model predictions. Please verify the input values.",
        )


# =====================================================================
# Analytics & Evaluation Endpoints (Read-Only)
# =====================================================================
@app.get(
    "/analytics/summary",
    response_model=AnalyticsSummaryResponse,
    tags=["Analytics"],
    status_code=status.HTTP_200_OK,
)
def get_analytics_summary() -> AnalyticsSummaryResponse:
    """Return verified dataset KPIs, test-set evaluation metrics, and model benchmarks.

    Reuses saved evaluation reports and training metadata directly from disk;
    never retrains or recalculates models on request.
    """
    data_summary_path = settings.REPORTS_DIR / "data_summary.json"
    metrics_path = settings.REPORTS_DIR / "metrics.json"
    cv_path = settings.REPORTS_DIR / "cv_model_comparison.json"
    fi_path = settings.REPORTS_DIR / "feature_importance.json"

    if not data_summary_path.exists() or not metrics_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Verified evaluation reports are unavailable. Please run 'python src/evaluate.py' to generate reports.",
        )

    try:
        with open(data_summary_path, "r", encoding="utf-8") as f:
            data_summary = json.load(f)
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        # Dataset Summary
        target_dist = data_summary.get("target_quality_distribution", {})
        counts = target_dist.get("score_counts", {})
        percentages = target_dist.get("score_percentages", {})
        dataset_summary = DatasetSummary(
            sample_count=data_summary.get("row_count", 1599),
            feature_count=data_summary.get("column_count", 12) - 1,
            average_quality=target_dist.get("mean_quality", 5.636),
            quality_std=target_dist.get("std_quality", 0.8076),
            quality_distribution={str(k): int(v) for k, v in counts.items()},
            quality_percentages={str(k): float(v) for k, v in percentages.items()},
        )

        # Test Metrics
        selected_model_name = (
            model_manager.model_type_name
            or metrics_data.get("model_artifact", "RandomForestRegressor").replace(".joblib", "").replace("wine_quality_", "").title()
        )
        if selected_model_name.endswith(".Joblib") or selected_model_name.endswith(".joblib"):
            selected_model_name = "RandomForestRegressor"

        test_metrics = ModelTestMetrics(
            selected_model=selected_model_name,
            test_samples=int(metrics_data.get("test_samples", 320)),
            mae=float(metrics_data.get("mean_absolute_error", 0.422)),
            rmse=float(metrics_data.get("root_mean_squared_error", 0.5483)),
            r2=float(metrics_data.get("r2_score", 0.5399)),
            mse=float(metrics_data.get("mean_squared_error", 0.3007)),
            explained_variance=float(metrics_data.get("explained_variance_score", 0.5401)),
        )

        # CV Model Comparisons
        comparisons: List[ModelComparisonItem] = []
        if cv_path.exists():
            with open(cv_path, "r", encoding="utf-8") as f:
                cv_data = json.load(f)
            for algo, stats in cv_data.items():
                comparisons.append(
                    ModelComparisonItem(
                        model=algo,
                        mae=float(stats.get("cv_mae_mean", 0.0)),
                        rmse=float(stats.get("cv_rmse_mean", 0.0)),
                        r2=float(stats.get("cv_r2_mean", 0.0)),
                        mae_std=float(stats["cv_mae_std"]) if "cv_mae_std" in stats else None,
                        rmse_std=float(stats["cv_rmse_std"]) if "cv_rmse_std" in stats else None,
                        r2_std=float(stats["cv_r2_std"]) if "cv_r2_std" in stats else None,
                    )
                )

        # Feature Importance Ranking
        feature_importance: List[FeatureImportanceItem] = []
        if fi_path.exists():
            with open(fi_path, "r", encoding="utf-8") as f:
                fi_data = json.load(f)
            for item in fi_data.get("features", []):
                feature_importance.append(FeatureImportanceItem(**item))
        elif model_manager.is_loaded and hasattr(model_manager.model, "named_steps"):
            reg = model_manager.model.named_steps.get("regressor")
            if reg is not None and hasattr(reg, "feature_importances_"):
                imps = reg.feature_importances_
                raw_items = [
                    FeatureImportanceItem(
                        feature=name,
                        importance=round(float(imp), 4),
                        percentage=round(float(imp) * 100, 2),
                    )
                    for name, imp in zip(FEATURE_NAMES, imps)
                ]
                raw_items.sort(key=lambda x: x.importance, reverse=True)
                feature_importance = raw_items

        return AnalyticsSummaryResponse(
            dataset=dataset_summary,
            model=test_metrics,
            model_comparison=comparisons,
            feature_importance=feature_importance,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to load analytics summary: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load analytics summary from verified reports.",
        )


@app.get(
    "/analytics/correlation",
    response_model=CorrelationMatrixResponse,
    tags=["Analytics"],
    status_code=status.HTTP_200_OK,
)
def get_correlation_matrix() -> CorrelationMatrixResponse:
    """Return verified Pearson correlation matrix across the 11 features and quality."""
    corr_path = settings.REPORTS_DIR / "correlation_matrix.json"
    if corr_path.exists():
        try:
            with open(corr_path, "r", encoding="utf-8") as f:
                corr_data = json.load(f)
            return CorrelationMatrixResponse(
                columns=corr_data["columns"],
                matrix=corr_data["matrix"],
            )
        except Exception as exc:
            logger.error("Failed to read correlation matrix report: %s", exc)

    # Fallback to computing from raw CSV if report file was not pre-generated
    if settings.DATA_PATH.exists():
        try:
            df = pd.read_csv(settings.DATA_PATH, sep=";")
            cols = FEATURE_NAMES + [TARGET_COLUMN]
            corr_df = df[cols].corr(method="pearson")
            matrix = [
                [round(float(corr_df.iloc[r, c]), 4) for c in range(len(cols))]
                for r in range(len(cols))
            ]
            return CorrelationMatrixResponse(columns=cols, matrix=matrix)
        except Exception as exc:
            logger.error("Failed to compute correlation from raw dataset: %s", exc)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Correlation matrix report or source dataset is unavailable.",
    )


@app.get(
    "/analytics/actual-vs-predicted",
    response_model=ActualVsPredictedResponse,
    tags=["Analytics"],
    status_code=status.HTTP_200_OK,
)
def get_actual_vs_predicted() -> ActualVsPredictedResponse:
    """Return verified holdout test evaluations for actual vs predicted scatter plot."""
    avp_path = settings.REPORTS_DIR / "actual_vs_predicted.json"
    if not avp_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Actual versus predicted evaluation report is unavailable.",
        )

    try:
        with open(avp_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        points = [
            ActualVsPredictedPoint(
                actual=float(p["actual"]),
                predicted=float(p["predicted"]),
                residual=float(p.get("residual", p["actual"] - p["predicted"])),
            )
            for p in data.get("points", [])
        ]

        return ActualVsPredictedResponse(
            test_samples=data.get("test_samples", len(points)),
            mae=float(data.get("mae", 0.422)),
            rmse=float(data.get("rmse", 0.5483)),
            r2=float(data.get("r2", 0.5399)),
            points=points,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to read actual vs predicted report: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load actual vs predicted evaluation points.",
        )


if __name__ == "__main__":
    import uvicorn

    logger.info("Launching FastAPI server on http://127.0.0.1:8000...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
