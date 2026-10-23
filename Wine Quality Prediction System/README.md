# Wine Quality Prediction System

A complete, production-grade Machine Learning system designed to predict the sensory quality score of red wine based on 11 physicochemical laboratory measurements.

The system is structured as a decoupled multi-tier architecture:
- **ML & Data Pipeline (Python 3.13)**: Preprocessing, model training, evaluation metrics, visual diagnostics, and CLI inference.
- **REST API Gateway (FastAPI)**: Validated Pydantic contracts, CORS-enabled endpoints, and real-time model inference.
- **Frontend Presentation Layer (Angular 21)**: Reactive UI with glassmorphic design, sensory presets, and instant feedback.

---

## 1. Project Directory Structure

```text
wine-quality-prediction/
├── ml-backend/
│   ├── data/
│   │   ├── .gitkeep
│   │   └── winequality-red.csv           # Semicolon-delimited source dataset
│   ├── notebooks/
│   │   └── exploratory_analysis.ipynb    # In-depth EDA notebook
│   ├── src/
│   │   ├── __init__.py                  # Package exports
│   │   ├── preprocessing.py             # Data loading, validation, and splitting
│   │   ├── train.py                     # Pipeline training and artifact saving
│   │   ├── evaluate.py                  # Regression metrics & diagnostic plots
│   │   └── predict.py                   # Prediction class and CLI runner
│   ├── models/
│   │   ├── .gitkeep
│   │   └── wine_quality_model.joblib    # Serialized scikit-learn pipeline
│   ├── reports/
│   │   ├── .gitkeep
│   │   ├── evaluation_report.md         # Comprehensive markdown project report
│   │   ├── metrics.json                 # Core regression metrics (MAE, MSE, RMSE, R²)
│   │   ├── evaluation_summary.json      # Complete evaluation metadata & residual stats
│   │   ├── hyperparameter_tuning.json   # Multi-model CV search results & comparisons
│   │   ├── actual_vs_predicted.png      # Parity scatter plot with 1:1 identity line
│   │   ├── residuals_distribution.png   # Residual histogram & KDE with Gaussian fit
│   │   ├── residuals_vs_predicted.png   # Homoscedasticity diagnostic scatter plot
│   │   ├── feature_importance.png       # Non-causal feature importance ranking
│   │   └── residuals_plot.png           # 2-panel composite residual plot
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_preprocessing.py        # Preprocessing & data integrity tests
│   │   ├── test_model.py                # Model training, evaluation & inference tests
│   │   └── test_api.py                  # FastAPI REST endpoint integration tests
│   ├── main.py                          # FastAPI gateway
│   └── requirements.txt                 # Pinned dependencies
├── wine-quality-ui/                     # Angular application
│   ├── src/
│   │   ├── app/
│   │   │   ├── models/wine.model.ts     # TypeScript data contracts
│   │   │   ├── services/wine-prediction.service.ts # HTTP API client
│   │   │   ├── app.ts                   # Component controller & presets
│   │   │   ├── app.html                 # Modern wine dashboard template
│   │   │   └── app.css                  # Dark glassmorphic styling
│   │   ├── index.html
│   │   └── styles.css
│   ├── angular.json
│   └── package.json
├── .gitignore
└── README.md
```

---

## 2. Dataset Specification

The project utilizes the **UCI Wine Quality Dataset (Red Wine)** collected by P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis (2009).

### Semicolon Separation
The source file is formatted with **semicolon (`;`) delimiters** rather than commas. The ingestion pipeline explicitly uses:
```python
pd.read_csv(filepath, sep=";")
```

### The 11 Input Features
| # | Feature Name | Unit / Description | Typical Red Range |
|---|---|---|---|
| 1 | `fixed acidity` | Tartaric acid ($g/dm^3$) | 4.6 - 15.9 |
| 2 | `volatile acidity` | Acetic acid ($g/dm^3$) (vinegar smell/defect) | 0.12 - 1.58 |
| 3 | `citric acid` | Citric acid ($g/dm^3$) (freshness/flavor) | 0.00 - 1.00 |
| 4 | `residual sugar` | Sugar remaining post-fermentation ($g/dm^3$) | 0.9 - 15.5 |
| 5 | `chlorides` | Sodium chloride / salt content ($g/dm^3$) | 0.012 - 0.611 |
| 6 | `free sulfur dioxide` | Free molecular $SO_2$ gas ($mg/dm^3$) | 1.0 - 72.0 |
| 7 | `total sulfur dioxide` | Free + bound forms of $SO_2$ ($mg/dm^3$) | 6.0 - 289.0 |
| 8 | `density` | Density of wine relative to water ($g/cm^3$) | 0.990 - 1.004 |
| 9 | `pH` | Wine acidity scale (3.0 to 4.0 typically) | 2.74 - 4.01 |
| 10 | `sulphates` | Potassium sulphate ($g/dm^3$) (antimicrobial) | 0.33 - 2.00 |
| 11 | `alcohol` | Alcohol volume percent (% vol) | 8.4 - 14.9 |

### Regression Target
- **`quality`**: Continuous/ordinal regression target scored by sensory experts from **0 (very poor)** to **10 (exceptional)**. In this dataset, scores range between **3 and 8** with a median of **6**.

---

## 3. Dataset Download Instructions

The source dataset is placed at `ml-backend/data/winequality-red.csv`. If you need to re-download or update it:

### Option A: Automated PowerShell Download (Recommended)
From the project root:
```powershell
curl.exe -L -o "ml-backend/data/winequality-red.csv" "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
```

Or using native PowerShell `Invoke-WebRequest`:
```powershell
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" -OutFile "ml-backend\data\winequality-red.csv"
```

### Option B: Manual Browser Download
1. Navigate to: `https://archive.ics.uci.edu/dataset/186/wine+quality`
2. Download `winequality-red.csv`.
3. Save or move the file into: `wine-quality-prediction/ml-backend/data/winequality-red.csv`.

---

## 4. Windows PowerShell Setup Guide

Follow these step-by-step commands in **Windows PowerShell**:

### Step 1: Navigate to the ML Backend
```powershell
cd "d:\COLLAGE CLASSES\SEM - 03\Machine Learning\Project\wine-quality-prediction\ml-backend"
```

### Step 2: Create a Python Virtual Environment
```powershell
python -m venv .venv
```

### Step 3: Configure Execution Policy and Activate Environment
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
*(You will see `(.venv)` in your terminal prompt).*

### Step 4: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Execution Workflow

All scripts utilize `pathlib.Path` and are runnable directly from `ml-backend`.

### 1. Data Preprocessing Verification
Inspect dataset loading, column validation, and train/test splitting:
```powershell
python src/preprocessing.py
```

### 2. Model Training
Fits the `StandardScaler` + `RandomForestRegressor` pipeline and persists `models/wine_quality_model.joblib`:
```powershell
python src/train.py
```
*Outputs holdout metrics (MSE, RMSE, MAE, R²) and writes `reports/train_metrics.json`.*

### 3. Model Evaluation, Diagnostic Plots & Report Generation
Evaluates the trained model artifact against the isolated holdout test set (320 samples, 20%), computes all regression metrics (MAE, MSE, RMSE, R²), produces all 4 diagnostic plots with non-causal associations disclaimers, and generates the Markdown project report:
```powershell
python src/evaluate.py
```

#### Optional Cross-Validated Hyperparameter Tuning
Tune Decision Tree and Random Forest hyperparameters strictly on the training partition using `GridSearchCV` or `RandomizedSearchCV`:
```powershell
# Tune both Decision Tree and Random Forest with GridSearchCV (5-fold CV)
python src/evaluate.py --tune --tune-model both --tune-strategy grid

# Fast exploration with RandomizedSearchCV (15 iterations)
python src/evaluate.py --tune --tune-model random_forest --tune-strategy random --n-iter 15
```

*Generated Evaluation Artifacts in `ml-backend/reports/`:*
- `reports/evaluation_report.md`: Detailed markdown project evaluation report explaining all metrics, diagnostic plots, tuning comparisons, and dataset limitations.
- `reports/metrics.json`: Actual measured test set regression metrics (MAE, MSE, RMSE, R²).
- `reports/evaluation_summary.json`: Complete evaluation metadata, dataset split, and residual statistics.
- `reports/hyperparameter_tuning.json`: Documented search spaces, CV scores, and single test re-evaluation.
- `reports/actual_vs_predicted.png`: Parity scatter plot with 1:1 identity line and metric annotations.
- `reports/residuals_distribution.png`: Error histogram and KDE with theoretical Gaussian overlay.
- `reports/residuals_vs_predicted.png`: Residuals against fitted values to detect heteroscedasticity.
- `reports/feature_importance.png`: Feature importance bar plot clearly labeled as model-dependent associations (non-causal).
- `reports/residuals_plot.png`: 2-panel composite residual figure.

### 4. Interactive Command-Line Inference
Test sample predictions via the CLI runner:
```powershell
python src/predict.py
```

### 5. Automated Unit & Integration Testing
Execute the complete test suite (39 tests covering preprocessing, training, hyperparameter tuning, evaluation, and REST API):
```powershell
pytest tests -v
```

### 6. Exploratory Data Analysis (EDA) Notebook
Launch Jupyter Notebook to view the complete analytical walkthrough:
```powershell
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## 6. Starting the FastAPI REST Server

Launch the Uvicorn web server hosting the prediction endpoints:
```powershell
uvicorn main:app --reload --port 8000
```

- **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Service Root Discovery**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Health Check Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Model Info & Training Metadata**: [http://127.0.0.1:8000/model-info](http://127.0.0.1:8000/model-info)
- **Feature Metadata**: [http://127.0.0.1:8000/features](http://127.0.0.1:8000/features)
- **Quality Prediction (POST)**: [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)

### Sample cURL Prediction Request:
```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{
    "fixed_acidity": 8.5,
    "volatile_acidity": 0.28,
    "citric_acid": 0.49,
    "residual_sugar": 2.1,
    "chlorides": 0.055,
    "free_sulfur_dioxide": 18.0,
    "total_sulfur_dioxide": 40.0,
    "density": 0.9952,
    "pH": 3.32,
    "sulphates": 0.82,
    "alcohol": 12.5
  }'
```

---

## 7. Starting the Angular Frontend

From a new PowerShell terminal, navigate to the `wine-quality-ui` directory:
```powershell
cd "d:\COLLAGE CLASSES\SEM - 03\Machine Learning\Project\wine-quality-prediction\wine-quality-ui"
npm install
npm start
```
Open [http://localhost:4200](http://localhost:4200) in your browser to interact with the Wine Quality Prediction dashboard.
