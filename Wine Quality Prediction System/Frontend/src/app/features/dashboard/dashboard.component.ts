import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { finalize } from 'rxjs';
import { AnalyticsService } from '../../core/services/analytics.service';
import { PredictionService } from '../../core/services/prediction.service';
import { AnalyticsSummaryResponse, BackendConnectionStatus, HealthResponse } from '../../core/models/wine.model';
import {
  CardComponent,
  ButtonComponent,
  AlertComponent,
  LoadingIndicatorComponent,
  EmptyStateComponent
} from '../../shared';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    CardComponent,
    ButtonComponent,
    AlertComponent,
    LoadingIndicatorComponent,
    EmptyStateComponent
  ],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent implements OnInit {
  isLoading = false;
  error = '';
  summary: AnalyticsSummaryResponse | null = null;
  backendStatus: BackendConnectionStatus = 'checking';
  lastHealth: HealthResponse | null = null;

  constructor(
    private analyticsService: AnalyticsService,
    private predictionService: PredictionService,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    this.predictionService.backendStatus$.subscribe((status) => {
      this.backendStatus = status;
      this.cdr.markForCheck();
    });

    this.predictionService.lastHealth$.subscribe((health) => {
      this.lastHealth = health;
      this.cdr.markForCheck();
    });

    this.loadDashboardSummary();
  }

  loadDashboardSummary(): void {
    this.isLoading = true;
    this.error = '';
    this.cdr.markForCheck();

    this.analyticsService.getAnalyticsSummary().pipe(
      finalize(() => {
        this.isLoading = false;
        this.cdr.markForCheck();
      })
    ).subscribe({
      next: (summary) => {
        this.summary = summary;
        this.cdr.markForCheck();
      },
      error: (err) => {
        this.summary = null;
        this.error =
          'Verified dataset statistics and model evaluation reports are currently unreachable. Please ensure the backend is running or static evaluation reports exist.';
        this.cdr.markForCheck();
      }
    });
  }

  get statusLabel(): string {
    switch (this.backendStatus) {
      case 'connected':
        return 'Backend Online';
      case 'degraded':
        return 'Backend Degraded';
      case 'disconnected':
        return 'Backend Offline';
      default:
        return 'Connecting...';
    }
  }

  get statusVariant(): 'success' | 'warning' | 'danger' | 'neutral' {
    switch (this.backendStatus) {
      case 'connected':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'disconnected':
        return 'danger';
      default:
        return 'neutral';
    }
  }
}
