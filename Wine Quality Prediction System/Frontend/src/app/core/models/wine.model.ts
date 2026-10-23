/**
 * Core Data Models & Data Contracts for VinoSense AI
 */

export interface WineFeatures {
  fixed_acidity: number;
  volatile_acidity: number;
  citric_acid: number;
  residual_sugar: number;
  chlorides: number;
  free_sulfur_dioxide: number;
  total_sulfur_dioxide: number;
  density: number;
  pH?: number;
  ph?: number;
  sulphates: number;
  alcohol: number;
}

/**
 * Strict 11-feature request payload matching FastAPI WineFeaturesInput schema
 */
export interface WinePredictionRequest {
  fixed_acidity: number;
  volatile_acidity: number;
  citric_acid: number;
  residual_sugar: number;
  chlorides: number;
  free_sulfur_dioxide: number;
  total_sulfur_dioxide: number;
  density: number;
  ph: number;
  sulphates: number;
  alcohol: number;
}

export interface WinePredictionResponse {
  predicted_quality: number;
  model_type: string;
  message: string;
  rounded_quality: number;
  quality_category: string;
  status: string;
  model_name: string;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | string;
  model_loaded: boolean;
  model_type?: string | null;
  version: string;
  feature_count: number;
  dataset_available: boolean;
}

export interface TrainingMetadata {
  cv_folds?: number;
  primary_selection_metric?: string;
  estimator_hyperparameters?: string;
  cross_validation_results?: {
    algorithm: string;
    cv_mae_mean: number;
    cv_mae_std: number;
    cv_rmse_mean: number;
    cv_rmse_std: number;
    cv_mse_mean?: number;
    cv_mse_std?: number;
    cv_r2_mean: number;
    cv_r2_std: number;
    avg_fit_time_seconds: number;
    avg_score_time_seconds: number;
    fold_mae_scores?: number[];
  };
  dataset_information?: {
    total_rows: number;
    train_samples: number;
    test_samples: number;
    test_size: number;
  };
  training_timestamp_utc?: string;
  random_seed?: number;
}

export interface ModelInfoResponse {
  selected_estimator: string;
  model_type: string;
  required_feature_names: string[];
  feature_count: number;
  target_column: string;
  pipeline_steps: string[];
  training_metadata: TrainingMetadata | null;
}

export interface FeaturesMetadataResponse {
  features: string[];
  feature_count: number;
  target: string;
  target_type: string;
  score_range: string;
}

export interface ApiInfoResponse {
  project: string;
  title: string;
  version: string;
  description: string;
  documentation: string;
  redoc: string;
  health_check: string;
  model_info: string;
  prediction_endpoint: string;
}

export interface PredictionHistoryItem {
  id: string;
  timestamp: string;
  features: WineFeatures;
  prediction: WinePredictionResponse;
  latencyMs?: number;
}

export type BackendConnectionStatus = 'checking' | 'connected' | 'degraded' | 'disconnected';

export interface DatasetSummary {
  sample_count: number;
  feature_count: number;
  average_quality: number;
  quality_std: number;
  quality_distribution: Record<string, number>;
  quality_percentages: Record<string, number>;
}

export interface ModelTestMetrics {
  selected_model: string;
  test_samples: number;
  mae: number;
  rmse: number;
  r2: number;
  mse: number;
  explained_variance: number;
}

export interface ModelComparisonItem {
  model: string;
  mae: number;
  rmse: number;
  r2: number;
  mae_std?: number;
  rmse_std?: number;
  r2_std?: number;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
  percentage: number;
}

export interface AnalyticsSummaryResponse {
  dataset: DatasetSummary;
  model: ModelTestMetrics;
  model_comparison: ModelComparisonItem[];
  feature_importance: FeatureImportanceItem[];
}

export interface CorrelationMatrixResponse {
  columns: string[];
  matrix: number[][];
}

export interface ActualVsPredictedPoint {
  actual: number;
  predicted: number;
  residual: number;
}

export interface ActualVsPredictedResponse {
  test_samples: number;
  mae: number;
  rmse: number;
  r2: number;
  points: ActualVsPredictedPoint[];
}
