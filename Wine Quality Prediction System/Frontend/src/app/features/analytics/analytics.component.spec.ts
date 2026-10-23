import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { of, throwError, Subject } from 'rxjs';
import { AnalyticsComponent } from './analytics.component';
import { AnalyticsService } from '../../core/services/analytics.service';
import {
  ActualVsPredictedResponse,
  AnalyticsSummaryResponse,
  CorrelationMatrixResponse
} from '../../core/models/wine.model';

describe('AnalyticsComponent', () => {
  let component: AnalyticsComponent;
  let fixture: ComponentFixture<AnalyticsComponent>;
  let analyticsService: AnalyticsService;

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
      { model: 'LinearRegression', mae: 0.5057, rmse: 0.6618, r2: 0.3226, mae_std: 0.0178 },
      { model: 'DecisionTreeRegressor', mae: 0.4777, rmse: 0.8076, r2: -0.0065, mae_std: 0.0303 },
      { model: 'RandomForestRegressor', mae: 0.4319, rmse: 0.6008, r2: 0.442, mae_std: 0.0203 }
    ],
    feature_importance: [
      { feature: 'alcohol', importance: 0.2709, percentage: 27.09 },
      { feature: 'sulphates', importance: 0.1484, percentage: 14.84 },
      { feature: 'volatile acidity', importance: 0.1115, percentage: 11.15 }
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
      { actual: 6, predicted: 5.46, residual: 0.54 },
      { actual: 5, predicted: 5.19, residual: -0.19 },
      { actual: 7, predicted: 6.82, residual: 0.18 }
    ]
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AnalyticsComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        AnalyticsService
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(AnalyticsComponent);
    component = fixture.componentInstance;
    analyticsService = TestBed.inject(AnalyticsService);
  });

  it('should create the analytics component', () => {
    vi.spyOn(analyticsService, 'getAnalyticsSummary').mockReturnValue(of(mockSummary));
    vi.spyOn(analyticsService, 'getCorrelationMatrix').mockReturnValue(of(mockCorrelation));
    vi.spyOn(analyticsService, 'getActualVsPredicted').mockReturnValue(of(mockActualVsPredicted));

    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('should render loading indicator when analytics data is loading', () => {
    const summarySubject = new Subject<AnalyticsSummaryResponse>();
    vi.spyOn(analyticsService, 'getAnalyticsSummary').mockReturnValue(summarySubject.asObservable());
    vi.spyOn(analyticsService, 'getCorrelationMatrix').mockReturnValue(new Subject<CorrelationMatrixResponse>());
    vi.spyOn(analyticsService, 'getActualVsPredicted').mockReturnValue(new Subject<ActualVsPredictedResponse>());

    fixture.detectChanges();
    expect(component.isLoading).toBe(true);

    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('app-loading-indicator')).toBeTruthy();
  });

  it('should render all 5 verified charts when reports load successfully', () => {
    vi.spyOn(analyticsService, 'getAnalyticsSummary').mockReturnValue(of(mockSummary));
    vi.spyOn(analyticsService, 'getCorrelationMatrix').mockReturnValue(of(mockCorrelation));
    vi.spyOn(analyticsService, 'getActualVsPredicted').mockReturnValue(of(mockActualVsPredicted));

    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;

    // 1. Chart 1: Wine quality distribution
    const distSection = compiled.querySelector('#chart-distribution');
    expect(distSection).toBeTruthy();
    expect(distSection?.querySelectorAll('.dist-bar').length).toBe(6);
    expect(distSection?.textContent).toContain('42.6%'); // Score 5

    // 2. Chart 2: Correlation Heatmap
    const corrSection = compiled.querySelector('#chart-correlation');
    expect(corrSection).toBeTruthy();
    expect(corrSection?.querySelectorAll('.heatmap-cell').length).toBe(9); // 3x3 mock matrix

    // 3. Chart 3: Actual vs Predicted Parity Plot
    const paritySection = compiled.querySelector('#chart-parity');
    expect(paritySection).toBeTruthy();
    expect(paritySection?.querySelector('.parity-line')).toBeTruthy();
    expect(paritySection?.querySelectorAll('.scatter-dot').length).toBe(3);

    // 4. Chart 4: Feature Importance Ranking
    const importanceSection = compiled.querySelector('#chart-importance');
    expect(importanceSection).toBeTruthy();
    expect(importanceSection?.querySelectorAll('.importance-row').length).toBe(3);
    expect(importanceSection?.textContent).toContain('alcohol');
    expect(importanceSection?.textContent).toContain('27.1%');
    expect(importanceSection?.textContent).toContain('Non-Causal Disclaimer');

    // 5. Chart 5: Regression Model Comparison
    const compSection = compiled.querySelector('#chart-comparison');
    expect(compSection).toBeTruthy();
    expect(compSection?.querySelectorAll('.model-card').length).toBe(3);
    expect(compSection?.textContent).toContain('Random Forest (Selected)');
    expect(compSection?.textContent).toContain('Selected Champion');
  });

  it('should display helpful empty states when individual datasets/reports are missing', () => {
    vi.spyOn(analyticsService, 'getAnalyticsSummary').mockReturnValue(of({
      ...mockSummary,
      feature_importance: [],
      model_comparison: []
    }));
    vi.spyOn(analyticsService, 'getCorrelationMatrix').mockReturnValue(of({ columns: [], matrix: [] }));
    vi.spyOn(analyticsService, 'getActualVsPredicted').mockReturnValue(of({ test_samples: 0, mae: 0, rmse: 0, r2: 0, points: [] }));

    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;

    // Correlation empty state
    expect(compiled.querySelector('#chart-correlation app-empty-state')).toBeTruthy();

    // Actual vs predicted empty state
    expect(compiled.querySelector('#chart-parity app-empty-state')).toBeTruthy();

    // Feature importance empty state
    expect(compiled.querySelector('#chart-importance app-empty-state')).toBeTruthy();

    // Model comparison empty state
    expect(compiled.querySelector('#chart-comparison app-empty-state')).toBeTruthy();
  });

  it('should display global error state and alert when summary fetch fails', () => {
    vi.spyOn(analyticsService, 'getAnalyticsSummary').mockReturnValue(
      throwError(() => new Error('Server unreachable'))
    );
    vi.spyOn(analyticsService, 'getCorrelationMatrix').mockReturnValue(of(mockCorrelation));
    vi.spyOn(analyticsService, 'getActualVsPredicted').mockReturnValue(of(mockActualVsPredicted));

    fixture.detectChanges();

    expect(component.summary).toBeNull();
    expect(component.error).toBeTruthy();

    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('app-empty-state')).toBeTruthy();
    expect(compiled.querySelector('app-alert')).toBeTruthy();
  });

  it('should allow filtering the correlation matrix to focus on quality target', () => {
    vi.spyOn(analyticsService, 'getAnalyticsSummary').mockReturnValue(of(mockSummary));
    vi.spyOn(analyticsService, 'getCorrelationMatrix').mockReturnValue(of(mockCorrelation));
    vi.spyOn(analyticsService, 'getActualVsPredicted').mockReturnValue(of(mockActualVsPredicted));

    fixture.detectChanges();

    expect(component.filteredRows.length).toBe(3);

    // Toggle to quality row only
    component.onlyQualityRow = true;
    expect(component.filteredRows).toEqual(['quality']);
  });
});
