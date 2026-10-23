import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { BehaviorSubject, Observable, catchError, tap, throwError, timeout } from 'rxjs';
import { environment } from '../../../environments/environment';
import {
  BackendConnectionStatus,
  FeaturesMetadataResponse,
  HealthResponse,
  ModelInfoResponse,
  PredictionHistoryItem,
  WineFeatures,
  WinePredictionRequest,
  WinePredictionResponse
} from '../models/wine.model';
import { HistoryService } from './history.service';

@Injectable({
  providedIn: 'root'
})
export class PredictionService {
  private readonly baseUrl = environment.apiUrl;

  // Reactive connection state management
  private readonly backendStatusSubject = new BehaviorSubject<BackendConnectionStatus>('checking');
  public readonly backendStatus$ = this.backendStatusSubject.asObservable();

  private readonly lastHealthSubject = new BehaviorSubject<HealthResponse | null>(null);
  public readonly lastHealth$ = this.lastHealthSubject.asObservable();

  constructor(
    private http: HttpClient,
    private historyService: HistoryService
  ) {
    // Initial health verification
    this.checkHealth().subscribe({
      next: () => {},
      error: () => {}
    });
  }

  /**
   * Observable stream of stored prediction history records.
   */
  public get history$(): Observable<PredictionHistoryItem[]> {
    return this.historyService.history$;
  }

  /**
   * Check FastAPI backend connectivity and model readiness.
   * Updates reactive connection status without throwing uncaught UI crashes.
   */
  checkHealth(): Observable<HealthResponse> {
    this.backendStatusSubject.next('checking');
    return this.http.get<HealthResponse>(`${this.baseUrl}/health`).pipe(
      timeout(6000),
      tap((health) => {
        this.lastHealthSubject.next(health);
        if (health.model_loaded) {
          this.backendStatusSubject.next('connected');
        } else {
          this.backendStatusSubject.next('degraded');
        }
      }),
      catchError((error: any) => {
        this.lastHealthSubject.next(null);
        this.backendStatusSubject.next('disconnected');
        return throwError(() => error);
      })
    );
  }

  /**
   * Fetch active estimator architecture, required feature ordering, and training metadata.
   */
  getModelInfo(): Observable<ModelInfoResponse> {
    return this.http.get<ModelInfoResponse>(`${this.baseUrl}/model-info`).pipe(
      timeout(6000),
      catchError((error: any) => {
        return throwError(() => error);
      })
    );
  }

  /**
   * Fetch the 11 feature names, target definition, and standard score bounds.
   */
  getFeaturesMetadata(): Observable<FeaturesMetadataResponse> {
    return this.http.get<FeaturesMetadataResponse>(`${this.baseUrl}/features`).pipe(
      timeout(6000),
      catchError((error: any) => {
        return throwError(() => error);
      })
    );
  }

  /**
   * Send 11 validated chemical attributes to the trained ML model for sensory score prediction.
   * Stores the successful outcome in local history for session tracking.
   * Failed requests (network errors, 422, 500) trigger catchError and are NOT recorded.
   */
  predictQuality(features: WineFeatures): Observable<WinePredictionResponse> {
    const startTime = performance.now();

    // Clean payload to ensure standard field naming for FastAPI WineFeaturesInput schema
    const payload: WinePredictionRequest = {
      fixed_acidity: Number(features.fixed_acidity),
      volatile_acidity: Number(features.volatile_acidity),
      citric_acid: Number(features.citric_acid),
      residual_sugar: Number(features.residual_sugar),
      chlorides: Number(features.chlorides),
      free_sulfur_dioxide: Number(features.free_sulfur_dioxide),
      total_sulfur_dioxide: Number(features.total_sulfur_dioxide),
      density: Number(features.density),
      ph: Number(features.ph !== undefined && features.ph !== null ? features.ph : features.pH),
      sulphates: Number(features.sulphates),
      alcohol: Number(features.alcohol)
    };

    return this.http.post<WinePredictionResponse>(`${this.baseUrl}/predict`, payload).pipe(
      timeout(10000),
      tap((result) => {
        const latencyMs = Math.round(performance.now() - startTime);
        this.historyService.addRecord(features, result, latencyMs);
      }),
      catchError((error: any) => {
        return throwError(() => error);
      })
    );
  }

  /**
   * Clear all recorded history.
   */
  clearHistory(): void {
    this.historyService.clearHistory();
  }

  /**
   * Get configured API base URL.
   */
  get apiBaseUrl(): string {
    return this.baseUrl;
  }

  /**
   * Get current connection status snapshot.
   */
  get currentStatus(): BackendConnectionStatus {
    return this.backendStatusSubject.value;
  }
}
