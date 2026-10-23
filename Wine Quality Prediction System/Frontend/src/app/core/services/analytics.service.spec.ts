import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { AnalyticsService } from './analytics.service';
import {
  ActualVsPredictedResponse,
  AnalyticsSummaryResponse,
  CorrelationMatrixResponse
} from '../models/wine.model';
import { environment } from '../../../environments/environment';

describe('AnalyticsService', () => {
  let service: AnalyticsService;
  let httpMock: HttpTestingController;

  const mockSummary: AnalyticsSummaryResponse = {
    dataset: {
      sample_count: 1599,
      feature_count: 11,
      average_quality: 5.636,
      quality_std: 0.8076,
      quality_distribution: { '3': 10, '4': 53, '5': 681, '6': 638, '7': 199, '8': 18 },
      quality_percentages: { '3': 0.63, '4': 3.31, '5': 42.59, '6': 39.9, '7': 12.45, '8': 1.13 }
    },
    model: {
      selected_model: 'RandomForestRegressor',
      test_samples: 320,
      mae: 0.422,
      rmse: 0.5483,
      r2: 0.5399,
      mse: 0.3007,
      explained_variance: 0.5401
    },
    model_comparison: [
      { model: 'LinearRegression', mae: 0.5057, rmse: 0.6618, r2: 0.3226 },
      { model: 'DecisionTreeRegressor', mae: 0.4777, rmse: 0.8076, r2: -0.0065 },
      { model: 'RandomForestRegressor', mae: 0.4319, rmse: 0.6008, r2: 0.442 }
    ],
    feature_importance: [
      { feature: 'alcohol', importance: 0.2709, percentage: 27.09 },
      { feature: 'sulphates', importance: 0.1484, percentage: 14.84 }
    ]
  };

  const mockCorrelation: CorrelationMatrixResponse = {
    columns: ['alcohol', 'volatile acidity', 'quality'],
    matrix: [
      [1.0, -0.2022, 0.4762],
      [-0.2022, 1.0, -0.3906],
      [0.4762, -0.3906, 1.0]
    ]
  };

  const mockActualVsPredicted: ActualVsPredictedResponse = {
    test_samples: 320,
    mae: 0.422,
    rmse: 0.5483,
    r2: 0.5399,
    points: [
      { actual: 6, predicted: 5.3, residual: 0.7 },
      { actual: 5, predicted: 5.22, residual: -0.22 }
    ]
  };

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        AnalyticsService
      ]
    });

    service = TestBed.inject(AnalyticsService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should fetch verified analytics summary from backend endpoint', () => {
    service.getAnalyticsSummary().subscribe((result) => {
      expect(result).toEqual(mockSummary);
      expect(result.dataset.sample_count).toBe(1599);
      expect(result.model.mae).toBe(0.422);
      expect(result.model.selected_model).toBe('RandomForestRegressor');
    });

    const req = httpMock.expectOne(`${environment.apiUrl}/analytics/summary`);
    expect(req.request.method).toBe('GET');
    req.flush(mockSummary);
  });

  it('should fallback to static reports when backend /analytics/summary returns error', () => {
    service.getAnalyticsSummary().subscribe((result) => {
      expect(result.dataset.sample_count).toBe(1599);
      expect(result.model.selected_model).toBe('RandomForestRegressor');
      expect(result.model.mae).toBe(0.422);
    });

    // 1. Backend fails
    const backendReq = httpMock.expectOne(`${environment.apiUrl}/analytics/summary`);
    backendReq.error(new ProgressEvent('Network error'), { status: 503, statusText: 'Service Unavailable' });

    // 2. Static reports forkJoin requests
    const dsReq = httpMock.expectOne('/reports/data_summary.json');
    dsReq.flush({
      row_count: 1599,
      column_count: 12,
      target_quality_distribution: {
        mean_quality: 5.636,
        std_quality: 0.8076,
        score_counts: { '3': 10, '5': 681 },
        score_percentages: { '3': 0.63, '5': 42.59 }
      }
    });

    const mReq = httpMock.expectOne('/reports/metrics.json');
    mReq.flush({
      test_samples: 320,
      mean_absolute_error: 0.422,
      root_mean_squared_error: 0.5483,
      r2_score: 0.5399,
      mean_squared_error: 0.3007,
      explained_variance_score: 0.5401
    });

    const cvReq = httpMock.expectOne('/reports/cv_model_comparison.json');
    cvReq.flush({
      LinearRegression: { cv_mae_mean: 0.5057, cv_rmse_mean: 0.6618, cv_r2_mean: 0.3226 }
    });

    const fiReq = httpMock.expectOne('/reports/feature_importance.json');
    fiReq.flush({
      features: [{ feature: 'alcohol', importance: 0.2709, percentage: 27.09 }]
    });
  });

  it('should fetch correlation matrix from backend', () => {
    service.getCorrelationMatrix().subscribe((result) => {
      expect(result.columns.length).toBe(3);
      expect(result.matrix[0][0]).toBe(1.0);
    });

    const req = httpMock.expectOne(`${environment.apiUrl}/analytics/correlation`);
    expect(req.request.method).toBe('GET');
    req.flush(mockCorrelation);
  });

  it('should fallback to /reports/correlation_matrix.json when backend fails', () => {
    service.getCorrelationMatrix().subscribe((result) => {
      expect(result.columns.length).toBe(3);
    });

    const backendReq = httpMock.expectOne(`${environment.apiUrl}/analytics/correlation`);
    backendReq.error(new ProgressEvent('Network error'), { status: 404 });

    const staticReq = httpMock.expectOne('/reports/correlation_matrix.json');
    staticReq.flush(mockCorrelation);
  });

  it('should fetch actual vs predicted points from backend', () => {
    service.getActualVsPredicted().subscribe((result) => {
      expect(result.test_samples).toBe(320);
      expect(result.points.length).toBe(2);
      expect(result.points[0].actual).toBe(6);
    });

    const req = httpMock.expectOne(`${environment.apiUrl}/analytics/actual-vs-predicted`);
    expect(req.request.method).toBe('GET');
    req.flush(mockActualVsPredicted);
  });
});
