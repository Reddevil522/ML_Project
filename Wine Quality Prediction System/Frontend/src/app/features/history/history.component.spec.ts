import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { BehaviorSubject } from 'rxjs';
import { HistoryComponent } from './history.component';
import { HistoryService } from '../../core/services/history.service';
import { PredictionHistoryItem, WineFeatures, WinePredictionResponse } from '../../core/models/wine.model';

describe('HistoryComponent', () => {
  let component: HistoryComponent;
  let fixture: ComponentFixture<HistoryComponent>;
  let historyService: HistoryService;
  let historySubject: BehaviorSubject<PredictionHistoryItem[]>;

  const sampleFeatures1: WineFeatures = {
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

  const sampleFeatures2: WineFeatures = {
    fixed_acidity: 8.5,
    volatile_acidity: 0.28,
    citric_acid: 0.56,
    residual_sugar: 1.8,
    chlorides: 0.092,
    free_sulfur_dioxide: 35.0,
    total_sulfur_dioxide: 103.0,
    density: 0.9969,
    ph: 3.30,
    sulphates: 0.75,
    alcohol: 10.5
  };

  const samplePrediction1: WinePredictionResponse = {
    predicted_quality: 5.25,
    model_type: 'RandomForestRegressor',
    message: 'Wine quality predicted successfully.',
    rounded_quality: 5,
    quality_category: 'Average / Standard',
    status: 'success',
    model_name: 'RandomForestRegressor'
  };

  const samplePrediction2: WinePredictionResponse = {
    predicted_quality: 7.15,
    model_type: 'RandomForestRegressor',
    message: 'Wine quality predicted successfully.',
    rounded_quality: 7,
    quality_category: 'Excellent / Premium',
    status: 'success',
    model_name: 'RandomForestRegressor'
  };

  const mockItems: PredictionHistoryItem[] = [
    {
      id: 'pred_newest_123',
      timestamp: '2026-10-22T10:30:00.000Z',
      features: sampleFeatures2,
      prediction: samplePrediction2,
      latencyMs: 38
    },
    {
      id: 'pred_older_456',
      timestamp: '2026-10-22T09:15:00.000Z',
      features: sampleFeatures1,
      prediction: samplePrediction1,
      latencyMs: 42
    }
  ];

  beforeEach(async () => {
    historySubject = new BehaviorSubject<PredictionHistoryItem[]>(mockItems);

    await TestBed.configureTestingModule({
      imports: [HistoryComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideRouter([]),
        {
          provide: HistoryService,
          useValue: {
            history$: historySubject.asObservable(),
            deleteRecord: vi.fn((id: string) => {
              const current = historySubject.value.filter((i) => i.id !== id);
              historySubject.next(current);
              return true;
            }),
            clearHistory: vi.fn(() => historySubject.next([])),
            downloadCsv: vi.fn(),
            getHistory: vi.fn(() => historySubject.value)
          }
        }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(HistoryComponent);
    component = fixture.componentInstance;
    historyService = TestBed.inject(HistoryService);
  });

  it('should create the history component and load records sorted newest first', () => {
    fixture.detectChanges();
    expect(component).toBeTruthy();
    expect(component.allRecords.length).toBe(2);
    expect(component.filteredRecords.length).toBe(2);
    expect(component.filteredRecords[0].id).toBe('pred_newest_123');
  });

  it('should filter records by search query (id, category, or score)', () => {
    component.searchQuery = 'Premium';
    component.applyFilters();
    fixture.detectChanges();
    expect(component.filteredRecords.length).toBe(1);
    expect(component.filteredRecords[0].prediction.quality_category).toBe('Excellent / Premium');

    component.searchQuery = '456';
    component.applyFilters();
    expect(component.filteredRecords.length).toBe(1);
    expect(component.filteredRecords[0].id).toBe('pred_older_456');

    component.clearSearch();
    expect(component.filteredRecords.length).toBe(2);
  });

  it('should filter records by quality category dropdown', () => {
    component.selectedCategory = 'Excellent / Premium';
    component.applyFilters();
    fixture.detectChanges();
    expect(component.filteredRecords.length).toBe(1);
    expect(component.filteredRecords[0].prediction.quality_category).toBe('Excellent / Premium');

    component.selectedCategory = 'Average / Standard';
    component.applyFilters();
    expect(component.filteredRecords.length).toBe(1);
    expect(component.filteredRecords[0].prediction.quality_category).toBe('Average / Standard');

    component.selectedCategory = 'ALL';
    component.applyFilters();
    expect(component.filteredRecords.length).toBe(2);
  });

  it('should sort records by predicted quality ascending and descending', () => {
    // Quality ascending (lowest first)
    component.currentSort = 'quality_asc';
    component.applyFilters();
    fixture.detectChanges();
    expect(component.filteredRecords[0].prediction.predicted_quality).toBe(5.25);
    expect(component.filteredRecords[1].prediction.predicted_quality).toBe(7.15);

    // Quality descending (highest first)
    component.currentSort = 'quality_desc';
    component.applyFilters();
    expect(component.filteredRecords[0].prediction.predicted_quality).toBe(7.15);
    expect(component.filteredRecords[1].prediction.predicted_quality).toBe(5.25);
  });

  it('should open inspection modal to examine all 11 physicochemical inputs', () => {
    component.inspectRecord(mockItems[0]);
    fixture.detectChanges();

    expect(component.selectedRecord).toEqual(mockItems[0]);

    const compiled = fixture.nativeElement as HTMLElement;
    const modal = compiled.querySelector('.modal-card');
    expect(modal).toBeTruthy();
    expect(modal?.textContent).toContain('pred_newest_123');
    expect(modal?.textContent).toContain('Fixed Acidity');
    expect(modal?.textContent).toContain('Alcohol Content');
    expect(modal?.textContent).toContain('10.5');

    // Close inspection
    component.closeInspection();
    expect(component.selectedRecord).toBeNull();
  });

  it('should delete an individual record', () => {
    component.deleteRecord('pred_newest_123');
    fixture.detectChanges();

    expect(historyService.deleteRecord).toHaveBeenCalledWith('pred_newest_123');
    expect(component.allRecords.length).toBe(1);
    expect(component.filteredRecords.length).toBe(1);
  });

  it('should show confirmation dialog and clear history when confirmed', () => {
    component.openClearConfirm();
    fixture.detectChanges();
    expect(component.showClearConfirm).toBe(true);

    component.confirmClear();
    expect(historyService.clearHistory).toHaveBeenCalled();
    expect(component.showClearConfirm).toBe(false);
    expect(component.allRecords.length).toBe(0);
  });

  it('should trigger CSV download on exportCsv click', () => {
    component.exportCsv();
    expect(historyService.downloadCsv).toHaveBeenCalled();
  });

  it('should display completely empty state when no records exist', () => {
    historySubject.next([]);
    fixture.detectChanges();

    expect(component.allRecords.length).toBe(0);
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.empty-state-section app-empty-state')).toBeTruthy();
    expect(compiled.textContent).toContain('No Prediction History Found');
  });

  it('should display filter empty state when search matches 0 records', () => {
    component.searchQuery = 'non_existent_query_xyz';
    component.applyFilters();
    fixture.detectChanges();

    expect(component.filteredRecords.length).toBe(0);
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.empty-filter-section app-empty-state')).toBeTruthy();
    expect(compiled.textContent).toContain('No Matching Records');

    // Reset filters
    component.resetFilters();
    expect(component.filteredRecords.length).toBe(2);
  });
});
