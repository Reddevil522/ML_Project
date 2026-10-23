"""Unit and integration tests for FastAPI REST endpoints.

Covers:
- Root and API discovery (GET /)
- Health check and model readiness (GET /health)
- Model architecture and training metadata (GET /model-info)
- Feature definitions and targets (GET /features)
- Valid prediction requests (POST /predict) with exact response structure
- Non-hardcoded prediction verification (distinct inputs produce distinct outputs)
- Case-insensitive pH alias handling ('pH' and 'ph')
- Invalid requests: missing fields, forbidden extra properties, out-of-bounds values, non-finite values (NaN, Inf)
- Domain cross-field physical consistency (free SO2 <= total SO2)
- Model-unavailable behavior (503 Service Unavailable when model is not loaded)
- CORS header configuration for Angular development origins
"""

from __future__ import annotations

import math
from typing import Any, Dict

from fastapi.testclient import TestClient
import pytest

from main import app, model_manager
from src.preprocessing import FEATURE_NAMES


@pytest.fixture(scope="module")
def client():
    """Module-scoped FastAPI TestClient running the application lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_wine_payload() -> Dict[str, Any]:
    """Fixture providing realistic red wine chemical attributes."""
    return {
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


# =====================================================================
# 1. Root & Discovery Tests
# =====================================================================
def test_api_root(client: TestClient) -> None:
    """Verify that root endpoint returns 200 and API discovery links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "Wine Quality Prediction System"
    assert "/docs" in data["documentation"]
    assert "/health" in data["health_check"]
    assert "/predict" in data["prediction_endpoint"]
    assert "/model-info" in data["model_info"]


# =====================================================================
# 2. Health & Status Tests
# =====================================================================
def test_api_health(client: TestClient) -> None:
    """Verify that /health reports healthy status and model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["dataset_available"] is True
    assert data["feature_count"] == 11
    assert data["model_type"] is not None


# =====================================================================
# 3. Model Info & Metadata Tests
# =====================================================================
def test_api_model_info(client: TestClient) -> None:
    """Verify that /model-info returns estimator details, exact feature order, and metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()

    # Estimator details
    assert "selected_estimator" in data
    assert data["selected_estimator"] in ["RandomForestRegressor", "DecisionTreeRegressor", "LinearRegression", "Ridge", "ElasticNet"]
    assert "model_type" in data
    assert len(data["model_type"]) > 0

    # Required feature names in exact column order
    assert data["required_feature_names"] == FEATURE_NAMES
    assert data["feature_count"] == 11
    assert data["target_column"] == "quality"
    assert "pipeline_steps" in data
    assert len(data["pipeline_steps"]) > 0

    # Training metadata
    assert data["training_metadata"] is not None
    assert "cross_validation_results" in data["training_metadata"]
    assert "dataset_information" in data["training_metadata"]


def test_api_features_metadata(client: TestClient) -> None:
    """Verify that /features describes all 11 features and the regression target."""
    response = client.get("/features")
    assert response.status_code == 200
    data = response.json()
    assert len(data["features"]) == 11
    assert data["features"] == FEATURE_NAMES
    assert data["target"] == "quality"
    assert data["target_type"] == "continuous_regression"


# =====================================================================
# 4. Prediction Endpoints - Valid Inputs & Response Structure
# =====================================================================
def test_api_predict_valid_sample(client: TestClient, valid_wine_payload: Dict[str, Any]) -> None:
    """Verify that /predict returns actual numeric output and expected response schema."""
    response = client.post("/predict", json=valid_wine_payload)
    assert response.status_code == 200
    data = response.json()

    # Core required fields per specification
    assert "predicted_quality" in data
    assert isinstance(data["predicted_quality"], float)
    assert not math.isnan(data["predicted_quality"])
    assert 3.0 <= data["predicted_quality"] <= 8.5

    assert "model_type" in data
    assert isinstance(data["model_type"], str)
    assert len(data["model_type"]) > 0

    assert "message" in data
    assert isinstance(data["message"], str)
    assert "predicted" in data["message"].lower()

    # UI backward compatibility fields
    assert data["status"] == "success"
    assert isinstance(data["rounded_quality"], int)
    assert data["quality_category"] in [
        "Excellent / Premium",
        "Good Quality",
        "Average / Standard",
        "Low / Deficient",
    ]


def test_api_predict_distinct_samples_not_hardcoded(client: TestClient) -> None:
    """Verify that prediction scores vary with input and are not hardcoded."""
    sample_low = {
        "fixed_acidity": 6.5,
        "volatile_acidity": 1.20,  # Very high volatile acidity (vinegar defect)
        "citric_acid": 0.00,
        "residual_sugar": 1.5,
        "chlorides": 0.150,
        "free_sulfur_dioxide": 6.0,
        "total_sulfur_dioxide": 20.0,
        "density": 0.9985,
        "ph": 3.65,
        "sulphates": 0.40,
        "alcohol": 9.0,  # Low alcohol
    }
    sample_high = {
        "fixed_acidity": 8.5,
        "volatile_acidity": 0.28,  # Clean, low volatile acidity
        "citric_acid": 0.45,
        "residual_sugar": 2.2,
        "chlorides": 0.065,
        "free_sulfur_dioxide": 18.0,
        "total_sulfur_dioxide": 40.0,
        "density": 0.9950,
        "ph": 3.30,
        "sulphates": 0.82,  # Good preservative
        "alcohol": 12.8,  # High alcohol
    }

    resp_low = client.post("/predict", json=sample_low)
    resp_high = client.post("/predict", json=sample_high)

    assert resp_low.status_code == 200
    assert resp_high.status_code == 200

    pred_low = resp_low.json()["predicted_quality"]
    pred_high = resp_high.json()["predicted_quality"]

    # Verify predictions are not identical (real inference on different features)
    assert pred_low != pred_high
    assert pred_high > pred_low, "High-quality wine sample should score higher than flawed sample"


def test_api_predict_ph_alias_support(client: TestClient, valid_wine_payload: Dict[str, Any]) -> None:
    """Verify that both 'ph' and 'pH' field aliases are accepted seamlessly."""
    payload_with_capital_ph = dict(valid_wine_payload)
    payload_with_capital_ph.pop("ph")
    payload_with_capital_ph["pH"] = 3.51

    response = client.post("/predict", json=payload_with_capital_ph)
    assert response.status_code == 200
    assert "predicted_quality" in response.json()


# =====================================================================
# 5. Input Validation & Defensible Bounds Rejection
# =====================================================================
def test_api_predict_invalid_missing_feature(client: TestClient) -> None:
    """Verify that /predict returns 422 when required fields are omitted."""
    invalid_payload = {
        "fixed_acidity": 7.4,
        "alcohol": 9.4,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "errors" in data


def test_api_predict_invalid_extra_field_forbidden(
    client: TestClient, valid_wine_payload: Dict[str, Any]
) -> None:
    """Verify that /predict strictly rejects unexpected/extra properties (extra='forbid')."""
    payload_with_extra = dict(valid_wine_payload)
    payload_with_extra["vintage"] = 2020
    payload_with_extra["region"] = "Bordeaux"

    response = client.post("/predict", json=payload_with_extra)
    assert response.status_code == 422
    data = response.json()
    field_errors = [err["field"] for err in data.get("errors", [])]
    assert any("vintage" in f for f in field_errors) or any("region" in f for f in field_errors)


def test_api_predict_invalid_out_of_bounds_alcohol(
    client: TestClient, valid_wine_payload: Dict[str, Any]
) -> None:
    """Verify that values outside defensible alcohol bounds [6.0, 20.0] are rejected."""
    # Exceeding upper bound
    payload_high = dict(valid_wine_payload)
    payload_high["alcohol"] = 35.0
    resp_high = client.post("/predict", json=payload_high)
    assert resp_high.status_code == 422

    # Below lower bound
    payload_low = dict(valid_wine_payload)
    payload_low["alcohol"] = 2.0
    resp_low = client.post("/predict", json=payload_low)
    assert resp_low.status_code == 422


def test_api_predict_invalid_out_of_bounds_acidity(
    client: TestClient, valid_wine_payload: Dict[str, Any]
) -> None:
    """Verify that values outside defensible fixed acidity bounds [3.0, 25.0] are rejected."""
    payload = dict(valid_wine_payload)
    payload["fixed_acidity"] = 1.0  # Physically impossible for wine
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_predict_invalid_out_of_bounds_ph(
    client: TestClient, valid_wine_payload: Dict[str, Any]
) -> None:
    """Verify that values outside defensible pH bounds [2.5, 4.5] are rejected."""
    payload = dict(valid_wine_payload)
    payload["ph"] = 6.8  # Neutral / non-acidic, not wine
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_predict_invalid_non_finite_values(
    client: TestClient, valid_wine_payload: Dict[str, Any]
) -> None:
    """Verify that NaN and Infinity numeric values are strictly rejected."""
    # Test string representations that would coerce to non-finite floats
    payload_nan = dict(valid_wine_payload)
    payload_nan["alcohol"] = "NaN"
    resp_nan = client.post("/predict", json=payload_nan)
    assert resp_nan.status_code == 422
    data_nan = resp_nan.json()
    assert any("finite" in str(err).lower() or "alcohol" in str(err).lower() for err in data_nan.get("errors", []))

    payload_inf = dict(valid_wine_payload)
    payload_inf["density"] = "Infinity"
    resp_inf = client.post("/predict", json=payload_inf)
    assert resp_inf.status_code == 422

    payload_neginf = dict(valid_wine_payload)
    payload_neginf["ph"] = "-Infinity"
    resp_neginf = client.post("/predict", json=payload_neginf)
    assert resp_neginf.status_code == 422


def test_api_predict_inconsistent_sulfur_dioxide(
    client: TestClient, valid_wine_payload: Dict[str, Any]
) -> None:
    """Verify that free SO2 exceeding total SO2 raises a 422 physical inconsistency error."""
    payload = dict(valid_wine_payload)
    payload["free_sulfur_dioxide"] = 50.0
    payload["total_sulfur_dioxide"] = 20.0  # Free cannot exceed total

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert any("free_sulfur_dioxide" in str(err) for err in data.get("errors", []))


# =====================================================================
# 6. Model-Unavailable Behavior
# =====================================================================
def test_api_model_unavailable_behavior(
    client: TestClient, valid_wine_payload: Dict[str, Any]
) -> None:
    """Verify that 503 is returned cleanly when model artifact is unavailable in memory."""
    # Temporarily stash and detach model
    saved_model = model_manager.model
    saved_type = model_manager.model_type_name
    try:
        model_manager.model = None
        model_manager.model_type_name = None

        # GET /health should report degraded
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "degraded"
        assert health_data["model_loaded"] is False

        # GET /model-info should return 503 Service Unavailable
        info_resp = client.get("/model-info")
        assert info_resp.status_code == 503
        assert "not loaded" in info_resp.json()["detail"].lower()

        # POST /predict should return 503 Service Unavailable
        pred_resp = client.post("/predict", json=valid_wine_payload)
        assert pred_resp.status_code == 503
        assert "not available" in pred_resp.json()["detail"].lower()
    finally:
        # Restore model
        model_manager.model = saved_model
        model_manager.model_type_name = saved_type

    # Verify recovery
    recovered_health = client.get("/health").json()
    assert recovered_health["status"] == "healthy"
    assert recovered_health["model_loaded"] is True


# =====================================================================
# 7. CORS Integration Tests
# =====================================================================
def test_api_cors_headers(client: TestClient) -> None:
    """Verify CORS middleware permits Angular frontend development origin (http://localhost:4200)."""
    headers = {
        "Origin": "http://localhost:4200",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    response = client.options("/predict", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:4200"


# =====================================================================
# 8. Analytics & Report Endpoints Tests
# =====================================================================
def test_analytics_summary_endpoint(client: TestClient) -> None:
    """Verify /analytics/summary returns verified dataset KPIs, test metrics, and CV comparison."""
    response = client.get("/analytics/summary")
    assert response.status_code == 200
    data = response.json()

    # Dataset KPIs
    assert data["dataset"]["sample_count"] == 1599
    assert data["dataset"]["feature_count"] == 11
    assert abs(data["dataset"]["average_quality"] - 5.636) < 0.01
    assert "5" in data["dataset"]["quality_distribution"]
    assert "6" in data["dataset"]["quality_distribution"]

    # Model test metrics
    assert data["model"]["selected_model"] == "RandomForestRegressor"
    assert data["model"]["test_samples"] == 320
    assert abs(data["model"]["mae"] - 0.422) < 0.001
    assert abs(data["model"]["rmse"] - 0.5483) < 0.001
    assert abs(data["model"]["r2"] - 0.5399) < 0.001

    # Model comparisons
    assert len(data["model_comparison"]) >= 3
    model_names = [m["model"] for m in data["model_comparison"]]
    assert "LinearRegression" in model_names
    assert "DecisionTreeRegressor" in model_names
    assert "RandomForestRegressor" in model_names

    # Feature importances
    assert len(data["feature_importance"]) == 11
    top_feature = data["feature_importance"][0]
    assert top_feature["feature"] == "alcohol"
    assert top_feature["importance"] > 0.20


def test_analytics_correlation_endpoint(client: TestClient) -> None:
    """Verify /analytics/correlation returns 12x12 Pearson correlation matrix."""
    response = client.get("/analytics/correlation")
    assert response.status_code == 200
    data = response.json()

    assert len(data["columns"]) == 12
    assert "alcohol" in data["columns"]
    assert "quality" in data["columns"]
    assert len(data["matrix"]) == 12
    for row in data["matrix"]:
        assert len(row) == 12
    # Diagonal correlation of feature with itself is 1.0
    for i in range(12):
        assert abs(data["matrix"][i][i] - 1.0) < 0.001


def test_analytics_actual_vs_predicted_endpoint(client: TestClient) -> None:
    """Verify /analytics/actual-vs-predicted returns holdout test evaluation points."""
    response = client.get("/analytics/actual-vs-predicted")
    assert response.status_code == 200
    data = response.json()

    assert data["test_samples"] == 320
    assert abs(data["mae"] - 0.422) < 0.001
    assert len(data["points"]) == 320
    first_pt = data["points"][0]
    assert "actual" in first_pt
    assert "predicted" in first_pt
    assert "residual" in first_pt
    assert 3 <= first_pt["actual"] <= 8
