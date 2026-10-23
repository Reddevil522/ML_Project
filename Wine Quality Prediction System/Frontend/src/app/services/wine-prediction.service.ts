import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { PredictionService } from '../core/services/prediction.service';
import { HealthResponse, WineFeatures, WinePredictionResponse } from '../core/models/wine.model';

/**
 * Backward compatibility wrapper delegating to core PredictionService.
 */
@Injectable({
  providedIn: 'root'
})
export class WinePredictionService {
  constructor(private coreService: PredictionService) {}

  getHealth(): Observable<HealthResponse> {
    return this.coreService.checkHealth();
  }

  predictQuality(features: WineFeatures): Observable<WinePredictionResponse> {
    return this.coreService.predictQuality(features);
  }

  getFeaturesMetadata(): Observable<any> {
    return this.coreService.getFeaturesMetadata();
  }
}
