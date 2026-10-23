import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { finalize } from 'rxjs';
import { AnalyticsService } from '../../core/services/analytics.service';
import {
  ActualVsPredictedPoint,
  ActualVsPredictedResponse,
  AnalyticsSummaryResponse,
  CorrelationMatrixResponse,
  FeatureImportanceItem,
  ModelComparisonItem
} from '../../core/models/wine.model';
import {
  CardComponent,
  BadgeComponent,
  ButtonComponent,
  AlertComponent,
  LoadingIndicatorComponent,
  EmptyStateComponent
} from '../../shared';

interface DistributionBar {
  score: string;
  count: number;
  percentage: number;
  heightPx: number;
  label: string;
}

export type ChartTabId = 'all' | 'distribution' | 'correlation' | 'parity' | 'importance' | 'comparison';

export interface ChartTab {
  id: ChartTabId;
  label: string;
}

@Component({
  selector: 'app-analytics',
  standalone: true,
  imports: [
    CommonModule,
    CardComponent,
    BadgeComponent,
    ButtonComponent,
    AlertComponent,
    LoadingIndicatorComponent,
    EmptyStateComponent
  ],
  templateUrl: './analytics.component.html',
  styleUrl: './analytics.component.css'
})
export class AnalyticsComponent implements OnInit {
  isLoading = false;
  isCorrLoading = false;
  isAvpLoading = false;
  error = '';

  summary: AnalyticsSummaryResponse | null = null;
  correlationData: CorrelationMatrixResponse | null = null;
  actualVsPredicted: ActualVsPredictedResponse | null = null;

  activeTab: ChartTabId = 'all';
  onlyQualityRow = false;

  activeDistTooltip: DistributionBar | null = null;
  activeCorrHover: { row: string; col: string; val: number } | null = null;
  activePointHover: { cx: number; cy: number; actual: number; predicted: number; residual: number } | null = null;

  chartTabs: ChartTab[] = [
    { id: 'all', label: 'All 5 Charts' },
    { id: 'distribution', label: '1. Quality Distribution' },
    { id: 'correlation', label: '2. Correlation Heatmap' },
    { id: 'parity', label: '3. Actual vs Predicted' },
    { id: 'importance', label: '4. Feature Importance' },
    { id: 'comparison', label: '5. Model Comparison' }
  ];

  constructor(
    private analyticsService: AnalyticsService,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    this.loadAllAnalytics();
  }

  loadAllAnalytics(): void {
    this.isLoading = true;
    this.isCorrLoading = true;
    this.isAvpLoading = true;
    this.error = '';
    this.cdr.markForCheck();

    // 1. Fetch summary (Distribution, Feature Importance, Model Comparison)
    this.analyticsService.getAnalyticsSummary().pipe(
      finalize(() => {
        this.isLoading = false;
        this.cdr.markForCheck();
      })
    ).subscribe({
      next: (sum) => {
        this.summary = sum;
        this.cdr.markForCheck();
      },
      error: (err) => {
        this.summary = null;
        this.error = 'Failed to load verified model analytics summary.';
        this.cdr.markForCheck();
      }
    });

    // 2. Fetch Pearson Correlation Matrix
    this.analyticsService.getCorrelationMatrix().pipe(
      finalize(() => {
        this.isCorrLoading = false;
        this.cdr.markForCheck();
      })
    ).subscribe({
      next: (corr) => {
        this.correlationData = corr;
        this.cdr.markForCheck();
      },
      error: () => {
        this.correlationData = null;
        this.cdr.markForCheck();
      }
    });

    // 3. Fetch Actual vs Predicted Points
    this.analyticsService.getActualVsPredicted().pipe(
      finalize(() => {
        this.isAvpLoading = false;
        this.cdr.markForCheck();
      })
    ).subscribe({
      next: (avp) => {
        this.actualVsPredicted = avp;
        this.cdr.markForCheck();
      },
      error: () => {
        this.actualVsPredicted = null;
        this.cdr.markForCheck();
      }
    });
  }

  // --- Helpers for Chart 1: Quality Distribution ---
  get distributionBars(): DistributionBar[] {
    if (!this.summary) return [];
    const dist = this.summary.dataset.quality_distribution;
    const pcts = this.summary.dataset.quality_percentages;
    const scores = ['3', '4', '5', '6', '7', '8'];

    return scores.map((sc) => {
      const count = dist[sc] || 0;
      const percentage = pcts[sc] || (count / this.summary!.dataset.sample_count) * 100;
      // Max count ~ 700, max height 180px
      const heightPx = Math.max(4, Math.round((count / 700) * 180));
      return {
        score: sc,
        count,
        percentage,
        heightPx,
        label: `Rating ${sc}`
      };
    });
  }

  getSensoryCategory(score: string): string {
    const s = parseInt(score, 10);
    if (s >= 7) return 'Premium Reserve Quality';
    if (s === 6) return 'Good Quality';
    if (s === 5) return 'Average / Standard Table Wine';
    return 'Low Quality / Deficient Sample';
  }

  // --- Helpers for Chart 2: Correlation Heatmap ---
  get filteredRows(): string[] {
    if (!this.correlationData) return [];
    if (this.onlyQualityRow) {
      return ['quality'];
    }
    return this.correlationData.columns;
  }

  getCorrVal(rowName: string, colName: string): number {
    if (!this.correlationData) return 0;
    const rIdx = this.correlationData.columns.indexOf(rowName);
    const cIdx = this.correlationData.columns.indexOf(colName);
    if (rIdx === -1 || cIdx === -1) return 0;
    return this.correlationData.matrix[rIdx][cIdx];
  }

  getHeatmapColor(val: number): string {
    if (val === 1.0) return 'rgba(163, 45, 84, 0.9)';
    if (val > 0) {
      const alpha = Math.min(0.85, Math.max(0.12, val * 1.2));
      return `rgba(163, 45, 84, ${alpha.toFixed(2)})`;
    }
    if (val < 0) {
      const alpha = Math.min(0.85, Math.max(0.12, Math.abs(val) * 1.2));
      return `rgba(32, 92, 135, ${alpha.toFixed(2)})`;
    }
    return 'rgba(255, 255, 255, 0.04)';
  }

  getHeatmapTextColor(val: number): string {
    if (Math.abs(val) > 0.35) return '#ffffff';
    return 'var(--text-secondary)';
  }

  formatFeatureShort(name: string): string {
    const map: Record<string, string> = {
      'fixed acidity': 'Fixed Acid',
      'volatile acidity': 'Vol Acid',
      'citric acid': 'Citric',
      'residual sugar': 'Sugar',
      'chlorides': 'Chlorides',
      'free sulfur dioxide': 'Free SO₂',
      'total sulfur dioxide': 'Tot SO₂',
      'density': 'Density',
      'pH': 'pH',
      'sulphates': 'Sulphates',
      'alcohol': 'Alcohol',
      'quality': '★ Quality'
    };
    return map[name] || name;
  }

  getCorrelationInterpretation(val: number): string {
    const absVal = Math.abs(val);
    const dir = val > 0 ? 'Positive' : 'Negative';
    if (absVal >= 0.7) return `Strong ${dir}`;
    if (absVal >= 0.4) return `Moderate ${dir}`;
    if (absVal >= 0.2) return `Weak ${dir}`;
    return 'Negligible';
  }

  // --- Helpers for Chart 3: Actual vs Predicted Parity Plot ---
  get parityPoints(): { cx: number; cy: number; actual: number; predicted: number; residual: number }[] {
    if (!this.actualVsPredicted) return [];
    // 320 test points
    // Plot area: x: 60 to 590 (width 530), y: 30 to 330 (height 300)
    // Actual domain: 3 to 8 (5 units)
    // Predicted range: ~3.8 to 8.2 (4.4 units)
    return this.actualVsPredicted.points.map((pt, i) => {
      // Deterministic jitter based on index to prevent discrete stacking
      const jitter = ((i % 17) - 8) * 0.015;
      const actualX = pt.actual + jitter;
      const cx = 60 + ((actualX - 3.0) / 5.0) * 530;
      const cy = 330 - ((pt.predicted - 4.0) / 4.0) * 300;
      return {
        cx: Math.max(60, Math.min(590, Math.round(cx * 10) / 10)),
        cy: Math.max(30, Math.min(330, Math.round(cy * 10) / 10)),
        actual: pt.actual,
        predicted: pt.predicted,
        residual: pt.residual
      };
    });
  }

  getResidualColor(residual: number): string {
    const absR = Math.abs(residual);
    if (absR < 0.3) return '#e5a93c'; // Gold (close match)
    if (absR < 0.6) return '#c44569'; // Rose (acceptable error)
    return '#ef4444'; // Red (larger error)
  }

  // --- Helpers for Chart 4: Feature Importance ---
  getImportanceWidth(percentage: number): number {
    // Max feature is ~27.1%, scale to 100% width
    return Math.min(100, Math.round((percentage / 28) * 100));
  }

  getFeatureRole(feature: string): string {
    const roles: Record<string, string> = {
      'fixed acidity': 'Structural crispness and tartness',
      'volatile acidity': 'Acetic acid indicator; spoilage aroma',
      'citric acid': 'Freshness enhancer and brightness',
      'residual sugar': 'Fermentation balance & mouthfeel',
      'chlorides': 'Saline mineral Terroir signature',
      'free sulfur dioxide': 'Active molecular preservative',
      'total sulfur dioxide': 'Combined bound + free SO₂',
      'density': 'Body weight relative to alcohol & sugar',
      'pH': 'Biological microbial stability',
      'sulphates': 'Antimicrobial salt additive',
      'alcohol': 'Primary sensory driver of perceived body'
    };
    return roles[feature] || 'Physicochemical predictor';
  }

  // --- Helpers for Chart 5: Model Comparison ---
  formatModelName(algo: string): string {
    switch (algo) {
      case 'LinearRegression':
        return 'Linear Regression';
      case 'DecisionTreeRegressor':
        return 'Decision Tree Regressor';
      case 'RandomForestRegressor':
        return 'Random Forest (Selected)';
      default:
        return algo;
    }
  }

  getMaeBarWidth(mae: number): number {
    // Lower is better, range ~0.40 to 0.55
    const ratio = Math.max(0, 1 - (mae - 0.40) / 0.20);
    return Math.round(ratio * 100);
  }

  getRmseBarWidth(rmse: number): number {
    const ratio = Math.max(0, 1 - (rmse - 0.55) / 0.30);
    return Math.round(ratio * 100);
  }

  getR2BarWidth(r2: number): number {
    if (r2 <= 0) return 4;
    return Math.min(100, Math.round(r2 * 200));
  }

  getModelRationale(model: string): string {
    switch (model) {
      case 'LinearRegression':
        return 'Fast parametric baseline; limited by linear additive assumptions over physicochemical interactions.';
      case 'DecisionTreeRegressor':
        return 'Prone to overfitting individual leaf partitions; negative CV R² indicates poor out-of-fold generalization.';
      case 'RandomForestRegressor':
        return '100-tree ensemble with bagging; reduces variance and handles nonlinear interactions with lowest CV MAE (0.4319).';
      default:
        return 'Candidate regression architecture.';
    }
  }
}
