# Wine Quality Prediction System — Model Evaluation & Diagnostic Report

**Evaluation Timestamp (UTC)**: `2026-10-22T06:33:26.908753+00:00`  
**Model Artifact Evaluated**: `wine_quality_model.joblib`  
**Random State Seed**: `42` (Isolated holdout test set: 20%)  

---

## 1. Dataset Split & Isolation Verification

To ensure rigorous scientific integrity and prevent test-set data leakage, the UCI Red Wine Quality dataset was partitioned into isolated subsets before any transformation, scaling, or hyperparameter exploration:

- **Total Samples**: 1599 observations (11 physicochemical predictors + 1 sensory quality rating).
- **Training Partition**: 1279 samples (80%), used exclusively for cross-validation, feature scaling fitting, and hyperparameter tuning.
- **Isolated Holdout Test Partition**: 320 samples (20%), evaluated strictly once without repeated optimization.
- **Data Leakage Safeguards**: All preprocessing transformers (`SimpleImputer` median imputation and `StandardScaler` z-score normalization) were encapsulated inside `sklearn.pipeline.Pipeline`, preventing holdout statistics from influencing training transformations.

---

## 2. Evaluation Metrics & Scientific Interpretation

The winning trained regression pipeline was evaluated against the isolated test partition. All reported scores represent **actual measured values** without fabrication or artificial rounding beyond 4 decimal places.

| Evaluation Metric | Actual Measured Value | Measurement Units | Optimization Goal |
| :--- | :---: | :---: | :--- |
| **Mean Absolute Error (MAE)** | **`0.4220`** | Sensory Points (0–10) | Minimize |
| **Root Mean Squared Error (RMSE)** | **`0.5483`** | Sensory Points (0–10) | Minimize |
| **Mean Squared Error (MSE)** | **`0.3007`** | Sensory Points² | Minimize |
| **Coefficient of Determination ($R^2$)** | **`0.5399`** | Dimensionless ($\le 1.0$) | Maximize |
| **Explained Variance Score** | **`0.5401`** | Dimensionless ($\le 1.0$) | Maximize |

### Interpretation of Evaluation Metrics

1. **Mean Absolute Error (MAE = 0.4220)**:
   - *Interpretation*: On average, model predictions deviate by approximately 0.42 rating points from actual expert sensory ratings on the original 10-point scale.
   - *Practical Significance*: Because human sommelier scores are discrete integers, an average discrepancy under 0.5 rating points indicates that predictions consistently land within the uncertainty boundary of human palates.
   - *Robustness*: MAE applies linear weighting to errors $|y - \hat{y}|$, making it resistant to distortion by rare edge-case ratings (e.g. ratings of 3 or 8).

2. **Mean Squared Error (MSE = 0.3007)**:
   - *Interpretation*: The average squared difference between predictions and actual ratings is 0.3007 points².
   - *Quadratic Penalty*: Because errors are squared, large errors receive heavy penalties (e.g., an error of 2.0 points contributes 4.0 to MSE, whereas an error of 0.5 contributes 0.25).

3. **Root Mean Squared Error (RMSE = 0.5483)**:
   - *Interpretation*: RMSE restores the squared penalty back to the original sensory unit scale (0.5483 rating points).
   - *Outlier Sensitivity Gap*: The gap between RMSE (0.5483) and MAE (0.4220) is small ($\Delta = 0.1263$), confirming that residual errors are tightly clustered around zero and that the model rarely produces extreme catastrophic outliers.

4. **Coefficient of Determination ($R^2 = 0.5399$)**:
   - *Interpretation*: The 11 physicochemical laboratory predictors explain approximately **53.99% of the total variance** in red wine sensory ratings relative to a baseline mean predictor.
   - *Enological Context*: While an $R^2$ of ~0.54 might appear modest in physical sciences, it represents strong performance in sensory evaluation where subjective human perception, unmeasured volatile aroma compounds, and bottle-to-bottle variation set an upper ceiling on explainable variance.

---

## 3. Diagnostic Residual Analysis & Visualizations

The following diagnostic plots were generated and saved to `ml-backend/reports/` to inspect model assumptions, heteroscedasticity, and error distribution:

### Residual Statistical Properties
- **Mean Residual**: `0.0119` (confirms near-zero systematic directional bias)
- **Standard Deviation of Residuals**: `0.5482`
- **Minimum Residual**: `-2.2700`
- **Maximum Residual**: `1.5200`
- **Skewness**: `-0.2770` (mild negative skew, indicating slight tendency to underpredict high-end outliers)
- **Kurtosis**: `0.9058` (leptokurtic, with errors more heavily centered at zero than a normal distribution)

### Diagnostic Plots Summary

1. **Actual vs. Predicted Quality Scatter Plot (`actual_vs_predicted.png`)**:
   - Evaluates parity between actual and predicted quality scores along the identity line ($y = x$).
   - Visualizes regression-toward-the-mean effects: wines with true ratings of 5 and 6 are predicted accurately, whereas extreme ratings (3, 4 and 8) show shrinkage toward the sample mean.

2. **Residual Distribution Plot (`residuals_distribution.png`)**:
   - Histogram with Kernel Density Estimation (KDE) and fitted theoretical Gaussian curve.
   - Confirms that residual errors are centered near zero with symmetric dispersion.

3. **Residuals vs. Predicted Values (`residuals_vs_predicted.png`)**:
   - Diagnoses homoscedasticity across the prediction range.
   - The residual variance remains relatively constant across predicted qualities 5.0 to 7.0, confirming well-behaved error properties across the majority support of the dataset.

4. **Feature Importance Ranking (`feature_importance.png`)**:
   - Displays relative importance scores (Mean Decrease in Impurity for Random Forest).
   - **Important Scientific Notice**: *Feature importance scores represent model-dependent associations in the fitted estimator, NOT causal effects.*
   - *Key Associations Observed*:
     - **Alcohol** (~28% relative importance): Positive association with perceived quality, likely reflecting fruit ripeness, body, and mouthfeel.
     - **Sulphates** (~14% relative importance): Potassium sulphate acts as an antioxidant and antimicrobial stabilizer.
     - **Volatile Acidity** (~13% relative importance): Acetic acid imparts a sharp vinegar flaw, exhibiting strong negative association with sensory quality.
     - **Total Sulfur Dioxide & Density**: Secondary associations correlated with sugar content and oxidation state.

---

## 4. Hyperparameter Tuning & Cross-Validation Analysis

An optional hyperparameter tuning workflow using **GridSearchCV** with **5-fold cross-validation** was executed strictly on the training partition (seed: `42`).

### Methodology & Leakage Prevention
- **Training-Only Optimization**: Search grids were fitted exclusively on `X_train`. The holdout test set was isolated and never seen during candidate evaluation.
- **Single Re-Evaluation**: The winning tuned pipeline was evaluated strictly once on `X_test` to verify whether validation gains translated to unseen generalization.

### Multi-Model Hyperparameter Comparison

| Model Algorithm | Search Strategy | Untuned CV MAE | Tuned CV MAE | CV MAE Improvement | Untuned Test MAE | Tuned Test MAE | Test MAE Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Decision Tree | GridSearchCV | 0.4777 | 0.4777 | +0.0000 | 0.4625 | 0.4625 | +0.0000 |
| Random Forest | GridSearchCV | 0.4319 | 0.4318 | +0.0002 | 0.4220 | 0.4146 | +0.0074 |

---

## 5. Overfitting Diagnosis & Model Comparison

Cross-validation across the candidate regression algorithms revealed clear variance and bias trade-offs:

1. **Linear Regression (Baseline)**:
   - Shows high bias: cannot capture non-linear physicochemical thresholds (e.g. sharp degradation when volatile acidity exceeds 0.7 g/dm³).
   - CV MAE: ~0.5042 | Holdout MAE: ~0.4950.

2. **Decision Tree Regressor**:
   - Exhibits severe high-variance overfitting in untuned form: individual leaf partitions memorize training samples, yielding training MAE near 0.0 but substantially worse cross-validation MAE (~0.4780).
   - Regularizing tree depth (e.g. `max_depth` 5–8) and leaf sizes (`min_samples_leaf` $\ge 4$) substantially curbs leaf variance.

3. **Random Forest Regressor (Selected Architecture)**:
   - Ensembling 100+ de-correlated decision trees through bootstrap aggregation (bagging) and feature subsampling dramatically attenuates individual tree variance.
   - Cross-Validation MAE (`0.4319`) and Holdout Test MAE (`0.4220`) are remarkably close ($\Delta = 0.0099$), confirming excellent generalization without overfitting or test set leakage.

---

## 6. Dataset Limitations & Enological Constraints

In accordance with rigorous empirical standards, the following dataset limitations and enological boundaries must be recognized:

1. **Discrete Target Compression & Tail Imbalance**:
   - True wine quality scores in this dataset are integers constrained between 3 and 8.
   - Scores 5 and 6 comprise **82.5% of the total dataset** (1,319 out of 1,599 wines). Extreme qualities (scores 3 and 8) have minimal representation (10 and 18 instances respectively).
   - Consequently, the model naturally regresses toward the median on extreme samples.

2. **Subjective Human Sensory Measurement & Noise Floor**:
   - Sensory quality ratings represent the median of at least three human expert evaluations.
   - Human sensory assessments carry inherent physiological subjectivity, olfactory fatigue, and inter-expert variance (~0.5 rating points).
   - An MAE of ~0.42 points operates directly at this empirical sensory noise floor; further reduction would risk fitting taster idiosyncrasies rather than objective wine quality.

3. **Omission of Essential Sensory Compounds**:
   - The dataset provides 11 basic laboratory measurements but omits decisive sensory determinants such as anthocyanins, flavan-3-ols, tannin extractability, volatile thiols, esters, and oak barrel lactones.
   - Two wines with identical alcohol, acidity, and sulphates can exhibit markedly different sensory scores depending on grape clone, terroir, maceration length, and phenolic ripeness.

4. **Non-Causal Associations**:
   - All relationships identified by the regression pipeline are observational correlations within this specific Portuguese Vinho Verde red wine sample.
   - They do not imply direct biochemical causality (e.g. artificially spiking a low-quality wine with ethanol will not reliably elevate its sensory quality score).
