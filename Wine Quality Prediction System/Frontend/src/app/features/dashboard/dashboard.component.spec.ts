import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { of, throwError, Subject } from 'rxjs';
import { DashboardComponent } from './dashboard.component';
import { AnalyticsService } from '../../core/services/analytics.service';
import { PredictionService } from '../../core/services/prediction.service';
import { AnalyticsSummaryResponse } from '../../core/models/wine.model';
import { provideRouter } from '@angular/router';

describe('DashboardComponent', () => {
  let component: DashboardComponent;
  let fixture: ComponentFixture<DashboardComponent>;
  let analyticsService: AnalyticsService;
  let predictionService: PredictionService;

  const mockVerifiedSummary: AnalyticsSummaryResponse = {
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
      { feature: 'alcohol', importance: 0.2709, percentage: 27.09 }
    ]
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        AnalyticsService,
        PredictionService
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(DashboardComponent);
    component = fixture.componentInstance;
    analyticsService = TestBed.inject(AnalyticsService);
    predictionService = TestBed.inject(PredictionService);
  });

  it('should create the dashboard component', () => {
    vi.spyOn(analyticsService, 'getAnalyticsSummary').mockReturnValue(of(mockVerifiedSummary));
    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('should render loading state while analytics summary is being fetched', () => {
    const summarySubject = new Subject<AnalyticsSummaryResponse>();
    vi.spyOn(analyticsService, 'getAnalyticsSummary').mockReturnValue(summarySubject.asObservable());

    fixture.detectChanges();
    expect(component.isLoading).toBe(true);

    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('app-loading-indicator')).toBeTruthy();
    expect(compiled.querySelector('#kpi-sample-count')).toBeNull();
  });

  it('should display verified KPI cards when analytics summary loads successfully', () => {
    vi.spyOn(analyticsService, 'getAnalyticsSummary').mockReturnValue(of(mockVerifiedSummary));

    fixture.detectChanges();

    expect(component.isLoading).toBe(false);
    expect(component.summary).toBeTruthy();

    const compiled = fixture.nativeElement as HTMLElement;

    // 1. Dataset sample count (1,599)
    const sampleCountEl = compiled.querySelector('#kpi-sample-count');
    expect(sampleCountEl).toBeTruthy();
    expect(sampleCountEl?.textContent?.trim()).toBe('1,599');

    // 2. Average observed quality (5.636 or 5.64)
    const avgQualityEl = compiled.querySelector('#kpi-avg-quality');
    expect(avgQualityEl).toBeTruthy();
    expect(avgQualityEl?.textContent?.trim()).toContain('5.636');

    // 3. Selected model (RandomForestRegressor)
    const selectedModelEl = compiled.querySelector('#kpi-selected-model');
    expect(selectedModelEl).toBeTruthy();
    expect(selectedModelEl?.textContent?.trim()).toBe('RandomForestRegressor');

    // 4. Test-set MAE, RMSE, and R²
    const maeEl = compiled.querySelector('#kpi-mae');
    const rmseEl = compiled.querySelector('#kpi-rmse');
    const r2El = compiled.querySelector('#kpi-r2');

    expect(maeEl?.textContent?.trim()).toBe('0.4220');
    expect(rmseEl?.textContent?.trim()).toBe('0.5483');
    expect(r2El?.textContent?.trim()).toBe('0.5399');
  });

  it('should display a helpful empty state and alert when reports are unavailable instead of fake KPIs', () => {
    vi.spyOn(analyticsService, 'getAnalyticsSummary').mockReturnValue(
      throwError(() => new Error('Reports unavailable'))
    );

    fixture.detectChanges();

    expect(component.isLoading).toBe(false);
    expect(component.summary).toBeNull();
    expect(component.error).toBeTruthy();

    const compiled = fixture.nativeElement as HTMLElement;

    // Must NOT render fake KPI values disguised as real results
    expect(compiled.querySelector('#kpi-sample-count')).toBeNull();
    expect(compiled.querySelector('#kpi-avg-quality')).toBeNull();
    expect(compiled.querySelector('#kpi-selected-model')).toBeNull();

    // Must render empty state with informative error
    const emptyStateEl = compiled.querySelector('app-empty-state');
    expect(emptyStateEl).toBeTruthy();
    expect(emptyStateEl?.textContent).toContain('Verified Metrics Unavailable');
  });

  it('should retry fetching reports when action button is clicked', () => {
    const spy = vi.spyOn(analyticsService, 'getAnalyticsSummary').mockReturnValue(
      throwError(() => new Error('Network error'))
    );

    fixture.detectChanges();
    expect(spy).toHaveBeenCalledTimes(1);

    // Call retry
    component.loadDashboardSummary();
    expect(spy).toHaveBeenCalledTimes(2);
  });
});
