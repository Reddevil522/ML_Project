import { TestBed } from '@angular/core/testing';
import { HistoryService, HISTORY_STORAGE_KEY } from './history.service';
import { WineFeatures, WinePredictionResponse } from '../models/wine.model';

describe('HistoryService', () => {
  let service: HistoryService;
  let mockStore: Record<string, string> = {};

  const mockLocalStorage = {
    getItem: vi.fn((key: string) => mockStore[key] ?? null),
    setItem: vi.fn((key: string, value: string) => {
      mockStore[key] = String(value);
    }),
    removeItem: vi.fn((key: string) => {
      delete mockStore[key];
    }),
    clear: vi.fn(() => {
      mockStore = {};
    }),
    get length() {
      return Object.keys(mockStore).length;
    },
    key: vi.fn((index: number) => Object.keys(mockStore)[index] ?? null)
  };

  const validFeatures: WineFeatures = {
    fixed_acidity: 7.4,
    volatile_acidity: 0.70,
    citric_acid: 0.00,
    residual_sugar: 1.9,
    chlorides: 0.076,
    free_sulfur_dioxide: 11.0,
    total_sulfur_dioxide: 34.0,
    density: 0.9978,
    ph: 3.51,
    sulphates: 0.56,
    alcohol: 9.4
  };

  const validPrediction: WinePredictionResponse = {
    predicted_quality: 5.6724,
    model_type: 'RandomForestRegressor',
    message: 'Wine quality predicted successfully.',
    rounded_quality: 6,
    quality_category: 'Average / Standard',
    status: 'success',
    model_name: 'RandomForestRegressor'
  };

  beforeEach(() => {
    mockStore = {};
    Object.defineProperty(globalThis, 'localStorage', {
      value: mockLocalStorage,
      writable: true,
      configurable: true
    });

    TestBed.configureTestingModule({
      providers: [HistoryService]
    });
    service = TestBed.inject(HistoryService);
  });

  afterEach(() => {
    mockStore = {};
  });

  it('should be created and initialize empty history when localStorage is empty', () => {
    expect(service).toBeTruthy();
    expect(service.getHistory()).toEqual([]);
  });

  it('should record a successful prediction with all 11 features and unique ID', () => {
    const record = service.addRecord(validFeatures, validPrediction, 45);

    expect(record).toBeTruthy();
    expect(record?.id).toContain('pred_');
    expect(record?.features.alcohol).toBe(9.4);
    expect(record?.prediction.predicted_quality).toBe(5.6724);
    expect(record?.latencyMs).toBe(45);

    const history = service.getHistory();
    expect(history.length).toBe(1);
    expect(history[0].id).toBe(record!.id);
  });

  it('should reject invalid or missing prediction payloads without corrupting history', () => {
    const badPrediction = { ...validPrediction, predicted_quality: NaN };
    const record = service.addRecord(validFeatures, badPrediction as any);

    expect(record).toBeNull();
    expect(service.getHistory().length).toBe(0);
  });

  it('should persist records in localStorage and preserve them across instances', () => {
    service.addRecord(validFeatures, validPrediction);

    const rawStored = mockLocalStorage.getItem(HISTORY_STORAGE_KEY);
    expect(rawStored).toBeTruthy();

    const parsed = JSON.parse(rawStored!);
    expect(parsed.length).toBe(1);
    expect(parsed[0].features.fixed_acidity).toBe(7.4);

    // Create a new service instance to verify rehydration from storage
    const newService = new HistoryService();
    expect(newService.getHistory().length).toBe(1);
    expect(newService.getHistory()[0].prediction.predicted_quality).toBe(5.6724);
  });

  it('should safely filter out malformed or corrupted records from localStorage', () => {
    const corruptedData = [
      { id: 'valid_1', timestamp: new Date().toISOString(), features: validFeatures, prediction: validPrediction },
      { id: 'corrupt_1', timestamp: 'not-a-date' }, // missing features
      { id: 'corrupt_2', timestamp: new Date().toISOString(), features: { ...validFeatures, alcohol: 'not-a-num' }, prediction: validPrediction },
      null,
      'a-random-string'
    ];

    mockStore[HISTORY_STORAGE_KEY] = JSON.stringify(corruptedData);

    const freshService = new HistoryService();
    const history = freshService.getHistory();

    expect(history.length).toBe(1);
    expect(history[0].id).toBe('valid_1');
  });

  it('should safely handle completely unparseable JSON in localStorage', () => {
    mockStore[HISTORY_STORAGE_KEY] = '{invalid-json';

    const freshService = new HistoryService();
    expect(freshService.getHistory()).toEqual([]);
  });

  it('should delete an individual record by id', () => {
    const r1 = service.addRecord(validFeatures, validPrediction);
    const r2 = service.addRecord({ ...validFeatures, alcohol: 12.5 }, { ...validPrediction, predicted_quality: 6.8 });

    expect(service.getHistory().length).toBe(2);

    const deleted = service.deleteRecord(r1!.id);
    expect(deleted).toBe(true);

    const remaining = service.getHistory();
    expect(remaining.length).toBe(1);
    expect(remaining[0].id).toBe(r2!.id);
  });

  it('should clear all history records and remove them from localStorage', () => {
    service.addRecord(validFeatures, validPrediction);
    expect(service.getHistory().length).toBe(1);

    service.clearHistory();
    expect(service.getHistory().length).toBe(0);
    expect(mockStore[HISTORY_STORAGE_KEY]).toBeUndefined();
  });

  it('should generate valid RFC 4180 CSV with proper headers and escaping', () => {
    const r1 = service.addRecord(validFeatures, {
      ...validPrediction,
      message: 'Result with, comma and "quotes"'
    });

    const csv = service.exportToCsv();

    // Headers
    expect(csv).toContain('Record ID');
    expect(csv).toContain('Predicted Quality');
    expect(csv).toContain('Fixed Acidity (g/dm3)');
    expect(csv).toContain('Alcohol (% vol)');

    // Values & Escaping
    expect(csv).toContain(r1!.id);
    expect(csv).toContain('5.6724');
    expect(csv).toContain('9.4');
  });

  it('should gracefully handle localStorage quota exceeded errors without data loss in memory', () => {
    mockLocalStorage.setItem.mockImplementationOnce(() => {
      const quotaError = new Error('Quota exceeded');
      quotaError.name = 'QuotaExceededError';
      throw quotaError;
    });

    const record = service.addRecord(validFeatures, validPrediction);
    expect(record).toBeTruthy();
    expect(service.getHistory().length).toBe(1);
  });
});
