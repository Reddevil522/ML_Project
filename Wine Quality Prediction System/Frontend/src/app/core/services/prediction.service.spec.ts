import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { PredictionService } from './prediction.service';
import { environment } from '../../../environments/environment';
import { WineFeatures, WinePredictionResponse, HealthResponse } from '../models/wine.model';

describe('PredictionService', () => {
  let service: PredictionService;
  let httpMock: HttpTestingController;

  const mockPredictionResponse: WinePredictionResponse = {
    predicted_quality: 5.6724,
    model_type: 'RandomForestRegressor',
    message: 'Wine quality predicted successfully.',
    rounded_quality: 6,
    quality_category: 'Average / Standard',
    status: 'success',
    model_name: 'RandomForestRegressor'
  };

  const sampleFeatures: WineFeatures = {
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

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        PredictionService
      ]
    });

    service = TestBed.inject(PredictionService);
    httpMock = TestBed.inject(HttpTestingController);

    // Handle initial constructor checkHealth call
    const initialHealthReq = httpMock.match(`${environment.apiUrl}/health`);
    initialHealthReq.forEach((req) =>
      req.flush({
        status: 'healthy',
        model_loaded: true,
        model_type: 'RandomForestRegressor',
        version: '1.0.0',
        feature_count: 11,
        dataset_available: true
      })
    );
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created and expose configured apiBaseUrl from environment', () => {
    expect(service).toBeTruthy();
    expect(service.apiBaseUrl).toBe(environment.apiUrl);
  });

  describe('predictQuality', () => {
    it('should send a POST request to /predict with exactly the 11 expected property names', () => {
      let actualResponse: WinePredictionResponse | undefined;

      service.predictQuality(sampleFeatures).subscribe((res) => {
        actualResponse = res;
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/predict`);
      expect(req.request.method).toBe('POST');

      // Verify payload property names and count
      const body = req.request.body;
      const expectedKeys = [
        'fixed_acidity',
        'volatile_acidity',
        'citric_acid',
        'residual_sugar',
        'chlorides',
        'free_sulfur_dioxide',
        'total_sulfur_dioxide',
        'density',
        'ph',
        'sulphates',
        'alcohol'
      ];

      expect(Object.keys(body).sort()).toEqual(expectedKeys.sort());
      expect(Object.keys(body).length).toBe(11);

      // Verify exact values and numeric types
      expect(body.fixed_acidity).toBe(7.4);
      expect(body.volatile_acidity).toBe(0.70);
      expect(body.citric_acid).toBe(0.00);
      expect(body.residual_sugar).toBe(1.9);
      expect(body.chlorides).toBe(0.076);
      expect(body.free_sulfur_dioxide).toBe(11.0);
      expect(body.total_sulfur_dioxide).toBe(34.0);
      expect(body.density).toBe(0.9978);
      expect(body.ph).toBe(3.51);
      expect(body.sulphates).toBe(0.56);
      expect(body.alcohol).toBe(9.4);

      req.flush(mockPredictionResponse);
      expect(actualResponse).toEqual(mockPredictionResponse);
      expect(actualResponse?.predicted_quality).toBe(5.6724);
      expect(actualResponse?.model_type).toBe('RandomForestRegressor');
    });

    it('should handle alternative pH alias and normalize to ph property name', () => {
      const featuresWithCapitalPH: WineFeatures = {
        ...sampleFeatures,
        pH: 3.42,
        ph: undefined
      };

      service.predictQuality(featuresWithCapitalPH).subscribe();

      const req = httpMock.expectOne(`${environment.apiUrl}/predict`);
      expect(req.request.body.ph).toBe(3.42);
      expect('pH' in req.request.body).toBe(false);
      req.flush(mockPredictionResponse);
    });

    it('should propagate 422 Unprocessable Entity validation errors from backend', () => {
      let capturedError: any;

      service.predictQuality(sampleFeatures).subscribe({
        next: () => {},
        error: (err) => {
          capturedError = err;
        }
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/predict`);
      req.flush(
        {
          detail: 'Input validation failed. Please correct the highlighted properties.',
          errors: [
            {
              field: 'fixed_acidity',
              message: 'Input should be greater than or equal to 3',
              type: 'greater_than_equal'
            }
          ]
        },
        { status: 422, statusText: 'Unprocessable Entity' }
      );

      expect(capturedError).toBeTruthy();
      expect(capturedError.status).toBe(422);
      expect(capturedError.error.errors.length).toBe(1);
      expect(capturedError.error.errors[0].field).toBe('fixed_acidity');
    });

    it('should propagate connection failure (status 0) when backend is offline', () => {
      let capturedError: any;

      service.predictQuality(sampleFeatures).subscribe({
        next: () => {},
        error: (err) => {
          capturedError = err;
        }
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/predict`);
      req.error(new ProgressEvent('error'), { status: 0, statusText: 'Unknown Error' });

      expect(capturedError).toBeTruthy();
      expect(capturedError.status).toBe(0);
    });

    it('should propagate 500 server errors when backend fails during evaluation', () => {
      let capturedError: any;

      service.predictQuality(sampleFeatures).subscribe({
        next: () => {},
        error: (err) => {
          capturedError = err;
        }
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/predict`);
      req.flush(
        { detail: 'An error occurred while evaluating model predictions.' },
        { status: 500, statusText: 'Internal Server Error' }
      );

      expect(capturedError).toBeTruthy();
      expect(capturedError.status).toBe(500);
      expect(capturedError.error.detail).toContain('evaluating model predictions');
    });
  });

  describe('checkHealth', () => {
    it('should update backendStatus to connected when model is loaded', () => {
      let healthRes: HealthResponse | undefined;

      service.checkHealth().subscribe((res) => {
        healthRes = res;
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/health`);
      req.flush({
        status: 'healthy',
        model_loaded: true,
        model_type: 'RandomForestRegressor',
        version: '1.0.0',
        feature_count: 11,
        dataset_available: true
      });

      expect(healthRes?.model_loaded).toBe(true);
      expect(service.currentStatus).toBe('connected');
    });

    it('should update backendStatus to degraded when model is not loaded', () => {
      service.checkHealth().subscribe();

      const req = httpMock.expectOne(`${environment.apiUrl}/health`);
      req.flush({
        status: 'degraded',
        model_loaded: false,
        model_type: null,
        version: '1.0.0',
        feature_count: 11,
        dataset_available: true
      });

      expect(service.currentStatus).toBe('degraded');
    });

    it('should update backendStatus to disconnected when health check fails', () => {
      service.checkHealth().subscribe({
        next: () => {},
        error: () => {}
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/health`);
      req.error(new ProgressEvent('error'), { status: 0 });

      expect(service.currentStatus).toBe('disconnected');
    });
  });
});
