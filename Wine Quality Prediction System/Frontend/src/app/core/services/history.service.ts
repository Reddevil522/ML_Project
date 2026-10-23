/**
 * HistoryService for VinoSense AI Wine Quality Prediction System
 *
 * NOTE FOR COLLEGE DEMONSTRATION & AUDIT:
 * Browser-local history is device- and browser-specific (stored via window.localStorage)
 * and is NOT a shared multi-user server database.
 * No secrets, tokens, or sensitive API keys are ever stored in localStorage.
 */

import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { PredictionHistoryItem, WineFeatures, WinePredictionResponse } from '../models/wine.model';

export const HISTORY_STORAGE_KEY = 'vinosense_prediction_history';
export const MAX_HISTORY_RECORDS = 100;

@Injectable({
  providedIn: 'root'
})
export class HistoryService {
  private readonly storageKey = HISTORY_STORAGE_KEY;
  private readonly historySubject = new BehaviorSubject<PredictionHistoryItem[]>([]);
  public readonly history$: Observable<PredictionHistoryItem[]> = this.historySubject.asObservable();

  constructor() {
    this.initHistory();
  }

  /**
   * Initialize history from browser localStorage with strict validation against corrupted data.
   */
  private initHistory(): void {
    const records = this.loadValidatedHistory();
    this.historySubject.next(records);
  }

  /**
   * Store a successful prediction result.
   *
   * Only real, validated predictions returned by the backend should be passed here.
   * Never store failed or mocked requests.
   */
  addRecord(
    features: WineFeatures,
    prediction: WinePredictionResponse,
    latencyMs?: number
  ): PredictionHistoryItem | null {
    if (!prediction || typeof prediction.predicted_quality !== 'number' || isNaN(prediction.predicted_quality)) {
      console.error('Cannot record history: Invalid prediction response received.');
      return null;
    }

    // Clean features to ensure all 11 physicochemical values are finite numbers
    const cleanFeatures: WineFeatures = {
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

    const newRecord: PredictionHistoryItem = {
      id: `pred_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
      timestamp: new Date().toISOString(),
      features: cleanFeatures,
      prediction: {
        predicted_quality: Number(prediction.predicted_quality),
        model_type: prediction.model_type || prediction.model_name || 'RandomForestRegressor',
        message: prediction.message || 'Wine quality predicted successfully.',
        rounded_quality: Number(prediction.rounded_quality ?? Math.round(prediction.predicted_quality)),
        quality_category: prediction.quality_category || this.inferQualityCategory(prediction.predicted_quality),
        status: 'success',
        model_name: prediction.model_name || prediction.model_type || 'RandomForestRegressor'
      },
      latencyMs: latencyMs !== undefined ? Math.round(latencyMs) : undefined
    };

    // Prepend to current items (newest first)
    const currentRecords = this.loadValidatedHistory();
    const updated = [newRecord, ...currentRecords].slice(0, MAX_HISTORY_RECORDS);

    this.historySubject.next(updated);
    this.persistHistory(updated);

    return newRecord;
  }

  /**
   * Delete an individual record by ID.
   */
  deleteRecord(id: string): boolean {
    const current = this.historySubject.value;
    const filtered = current.filter((item) => item.id !== id);

    if (filtered.length !== current.length) {
      this.historySubject.next(filtered);
      this.persistHistory(filtered);
      return true;
    }
    return false;
  }

  /**
   * Clear all prediction history records.
   */
  clearHistory(): void {
    this.historySubject.next([]);
    if (this.isLocalStorageAvailable()) {
      try {
        localStorage.removeItem(this.storageKey);
      } catch (e) {
        console.warn('Could not remove localStorage key:', e);
      }
    }
  }

  /**
   * Get synchronous snapshot of current records.
   */
  getHistory(): PredictionHistoryItem[] {
    return this.historySubject.value;
  }

  /**
   * Generate RFC 4180 compliant CSV string with proper escaping and column headers.
   */
  exportToCsv(records?: PredictionHistoryItem[]): string {
    const items = records || this.historySubject.value;

    const headers = [
      'Record ID',
      'Timestamp (UTC)',
      'Model Name',
      'Predicted Quality',
      'Rounded Quality',
      'Sensory Category',
      'Fixed Acidity (g/dm3)',
      'Volatile Acidity (g/dm3)',
      'Citric Acid (g/dm3)',
      'Residual Sugar (g/dm3)',
      'Chlorides (g/dm3)',
      'Free Sulfur Dioxide (mg/dm3)',
      'Total Sulfur Dioxide (mg/dm3)',
      'Density (g/cm3)',
      'pH',
      'Sulphates (g/dm3)',
      'Alcohol (% vol)',
      'Inference Latency (ms)'
    ];

    const escapeCsv = (val: any): string => {
      if (val === null || val === undefined) return '';
      const str = String(val);
      if (str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r')) {
        return `"${str.replace(/"/g, '""')}"`;
      }
      return str;
    };

    const rows: string[] = [headers.map(escapeCsv).join(',')];

    for (const r of items) {
      const feat = r.features;
      const pred = r.prediction;
      const phVal = feat.ph !== undefined ? feat.ph : feat.pH;

      const row = [
        escapeCsv(r.id),
        escapeCsv(r.timestamp),
        escapeCsv(pred.model_name || pred.model_type || 'RandomForestRegressor'),
        escapeCsv(pred.predicted_quality),
        escapeCsv(pred.rounded_quality),
        escapeCsv(pred.quality_category),
        escapeCsv(feat.fixed_acidity),
        escapeCsv(feat.volatile_acidity),
        escapeCsv(feat.citric_acid),
        escapeCsv(feat.residual_sugar),
        escapeCsv(feat.chlorides),
        escapeCsv(feat.free_sulfur_dioxide),
        escapeCsv(feat.total_sulfur_dioxide),
        escapeCsv(feat.density),
        escapeCsv(phVal),
        escapeCsv(feat.sulphates),
        escapeCsv(feat.alcohol),
        escapeCsv(r.latencyMs !== undefined ? r.latencyMs : '')
      ];
      rows.push(row.join(','));
    }

    return rows.join('\r\n');
  }

  /**
   * Programmatically download the CSV file in the browser.
   */
  downloadCsv(records?: PredictionHistoryItem[], filename?: string): void {
    const csvContent = this.exportToCsv(records);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const targetName =
      filename || `vinosense_prediction_history_${new Date().toISOString().slice(0, 10)}.csv`;

    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', targetName);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  /**
   * Load history from storage with schema validation.
   * Corrupted or non-compliant records are safely filtered out.
   */
  private loadValidatedHistory(): PredictionHistoryItem[] {
    if (!this.isLocalStorageAvailable()) {
      return [];
    }

    try {
      const raw = localStorage.getItem(this.storageKey);
      if (!raw) return [];

      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];

      // Validate each record before returning
      const validRecords: PredictionHistoryItem[] = [];
      for (const item of parsed) {
        if (this.isValidRecord(item)) {
          validRecords.push(item);
        }
      }

      // Sort newest first
      return validRecords.sort(
        (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
    } catch (e) {
      console.warn('Failed to parse history from localStorage, resetting corrupted storage:', e);
      return [];
    }
  }

  /**
   * Safely write records to localStorage with quota overflow recovery.
   */
  private persistHistory(items: PredictionHistoryItem[]): void {
    if (!this.isLocalStorageAvailable()) return;

    try {
      localStorage.setItem(this.storageKey, JSON.stringify(items));
    } catch (err: any) {
      if (err && (err.name === 'QuotaExceededError' || err.code === 22)) {
        console.warn('LocalStorage quota exceeded. Pruning older history to top 30.');
        try {
          const pruned = items.slice(0, 30);
          localStorage.setItem(this.storageKey, JSON.stringify(pruned));
        } catch {
          console.warn('LocalStorage remains full; history retained in memory.');
        }
      } else {
        console.warn('Failed to persist history to localStorage:', err);
      }
    }
  }

  /**
   * Validate schema of a history entry to prevent malformed data crashes.
   */
  private isValidRecord(item: any): item is PredictionHistoryItem {
    if (!item || typeof item !== 'object') return false;
    if (typeof item.id !== 'string' || !item.id) return false;
    if (typeof item.timestamp !== 'string' || isNaN(Date.parse(item.timestamp))) return false;

    // Validate features
    if (!item.features || typeof item.features !== 'object') return false;
    const f = item.features;
    const ph = f.ph !== undefined ? f.ph : f.pH;

    const requiredNumbers = [
      f.fixed_acidity,
      f.volatile_acidity,
      f.citric_acid,
      f.residual_sugar,
      f.chlorides,
      f.free_sulfur_dioxide,
      f.total_sulfur_dioxide,
      f.density,
      ph,
      f.sulphates,
      f.alcohol
    ];

    for (const val of requiredNumbers) {
      if (typeof val !== 'number' || isNaN(val)) return false;
    }

    // Validate prediction
    if (!item.prediction || typeof item.prediction !== 'object') return false;
    if (typeof item.prediction.predicted_quality !== 'number' || isNaN(item.prediction.predicted_quality)) {
      return false;
    }

    return true;
  }

  /**
   * Check if localStorage is supported and accessible in current environment.
   */
  private isLocalStorageAvailable(): boolean {
    try {
      if (typeof window === 'undefined' || typeof localStorage === 'undefined') {
        return false;
      }
      const test = '__storage_test__';
      localStorage.setItem(test, test);
      localStorage.removeItem(test);
      return true;
    } catch {
      return false;
    }
  }

  private inferQualityCategory(score: number): string {
    if (score >= 7.0) return 'Excellent / Premium';
    if (score >= 6.0) return 'Good Quality';
    if (score >= 5.0) return 'Average / Standard';
    return 'Low / Deficient';
  }
}
