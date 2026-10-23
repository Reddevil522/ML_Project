import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { of, throwError } from 'rxjs';
import { PredictionComponent, DATASET_BASELINE_VALUES } from './prediction.component';
import { PredictionService } from '../../core/services/prediction.service';
import { WinePredictionResponse } from '../../core/models/wine.model';

describe('PredictionComponent', () => {
  let component: PredictionComponent;
  let fixture: ComponentFixture<PredictionComponent>;
  let predictionService: PredictionService;

  const mockSuccessResponse: WinePredictionResponse = {
    predicted_quality: 6.2415,
    model_type: 'RandomForestRegressor',
    message: 'Wine quality predicted successfully.',
    rounded_quality: 6,
    quality_category: 'Good Quality',
    status: 'success',
    model_name: 'RandomForestRegressor'
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PredictionComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        PredictionService
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(PredictionComponent);
    component = fixture.componentInstance;
    predictionService = TestBed.inject(PredictionService);
  });

  it('should create the prediction component and initialize 11 form controls', () => {
    fixture.detectChanges();
    expect(component).toBeTruthy();
    expect(component.predictionForm).toBeTruthy();

    const expectedControls = [
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

    expectedControls.forEach((controlName) => {
      const control = component.predictionForm.get(controlName);
      expect(control).toBeTruthy();
      expect(control?.value).toBeDefined();
    });

    expect(Object.keys(component.predictionForm.controls).length).toBe(11);
    expect(component.predictionForm.valid).toBe(true);
  });

  describe('Form Validation', () => {
    it('should invalidate form when required fields are missing', () => {
      const control = component.predictionForm.get('fixed_acidity');
      control?.setValue(null);
      control?.markAsTouched();

      expect(control?.valid).toBe(false);
      expect(control?.errors?.['required']).toBe(true);
      expect(component.predictionForm.valid).toBe(false);
      expect(component.isFieldInvalid('fixed_acidity')).toBe(true);
    });

    it('should validate enological minimum bounds for all 11 attributes', () => {
      component.predictionForm.patchValue({
        fixed_acidity: 2.9, // min 3.0
        volatile_acidity: 0.04, // min 0.05
        citric_acid: -0.1, // min 0.0
        residual_sugar: 0.4, // min 0.5
        chlorides: 0.004, // min 0.005
        free_sulfur_dioxide: -1, // min 0
        total_sulfur_dioxide: -1, // min 0
        density: 0.9799, // min 0.9800
        ph: 2.49, // min 2.5
        sulphates: 0.09, // min 0.1
        alcohol: 5.9 // min 6.0
      });

      expect(component.predictionForm.get('fixed_acidity')?.errors?.['min']).toBeTruthy();
      expect(component.predictionForm.get('volatile_acidity')?.errors?.['min']).toBeTruthy();
      expect(component.predictionForm.get('citric_acid')?.errors?.['min']).toBeTruthy();
      expect(component.predictionForm.get('residual_sugar')?.errors?.['min']).toBeTruthy();
      expect(component.predictionForm.get('chlorides')?.errors?.['min']).toBeTruthy();
      expect(component.predictionForm.get('free_sulfur_dioxide')?.errors?.['min']).toBeTruthy();
      expect(component.predictionForm.get('total_sulfur_dioxide')?.errors?.['min']).toBeTruthy();
      expect(component.predictionForm.get('density')?.errors?.['min']).toBeTruthy();
      expect(component.predictionForm.get('ph')?.errors?.['min']).toBeTruthy();
      expect(component.predictionForm.get('sulphates')?.errors?.['min']).toBeTruthy();
      expect(component.predictionForm.get('alcohol')?.errors?.['min']).toBeTruthy();
      expect(component.predictionForm.valid).toBe(false);
    });

    it('should validate enological maximum bounds for all 11 attributes', () => {
      component.predictionForm.patchValue({
        fixed_acidity: 25.1, // max 25.0
        volatile_acidity: 3.1, // max 3.0
        citric_acid: 2.1, // max 2.0
        residual_sugar: 100.1, // max 100.0
        chlorides: 1.01, // max 1.0
        free_sulfur_dioxide: 151, // max 150
        total_sulfur_dioxide: 451, // max 450
        density: 1.0401, // max 1.0400
        ph: 4.51, // max 4.5
        sulphates: 3.01, // max 3.0
        alcohol: 20.1 // max 20.0
      });

      expect(component.predictionForm.get('fixed_acidity')?.errors?.['max']).toBeTruthy();
      expect(component.predictionForm.get('volatile_acidity')?.errors?.['max']).toBeTruthy();
      expect(component.predictionForm.get('citric_acid')?.errors?.['max']).toBeTruthy();
      expect(component.predictionForm.get('residual_sugar')?.errors?.['max']).toBeTruthy();
      expect(component.predictionForm.get('chlorides')?.errors?.['max']).toBeTruthy();
      expect(component.predictionForm.get('free_sulfur_dioxide')?.errors?.['max']).toBeTruthy();
      expect(component.predictionForm.get('total_sulfur_dioxide')?.errors?.['max']).toBeTruthy();
      expect(component.predictionForm.get('density')?.errors?.['max']).toBeTruthy();
      expect(component.predictionForm.get('ph')?.errors?.['max']).toBeTruthy();
      expect(component.predictionForm.get('sulphates')?.errors?.['max']).toBeTruthy();
      expect(component.predictionForm.get('alcohol')?.errors?.['max']).toBeTruthy();
      expect(component.predictionForm.valid).toBe(false);
    });

    it('should enforce that free SO2 cannot exceed total SO2', () => {
      component.predictionForm.patchValue({
        free_sulfur_dioxide: 50.0,
        total_sulfur_dioxide: 30.0
      });
      component.predictionForm.get('free_sulfur_dioxide')?.markAsTouched();

      expect(component.predictionForm.errors?.['freeSo2ExceedsTotal']).toBe(true);
      expect(component.hasSulfurDioxideError).toBe(true);
      expect(component.predictionForm.valid).toBe(false);

      // Resolving the condition makes it valid
      component.predictionForm.patchValue({
        free_sulfur_dioxide: 25.0,
        total_sulfur_dioxide: 50.0
      });

      expect(component.predictionForm.errors).toBeNull();
      expect(component.hasSulfurDioxideError).toBe(false);
      expect(component.predictionForm.valid).toBe(true);
    });
  });

  describe('Prediction Submission Workflow', () => {
    it('should prevent submission when form is invalid and mark controls touched', () => {
      const spy = vi.spyOn(predictionService, 'predictQuality');
      component.predictionForm.get('alcohol')?.setValue(null);

      component.onSubmit();

      expect(spy).not.toHaveBeenCalled();
      expect(component.predictionForm.get('alcohol')?.touched).toBe(true);
      expect(component.isLoading).toBe(false);
    });

    it('should prevent duplicate submission when already loading', () => {
      const spy = vi.spyOn(predictionService, 'predictQuality').mockReturnValue(of(mockSuccessResponse));
      component.isLoading = true;

      component.onSubmit();

      expect(spy).not.toHaveBeenCalled();
    });

    it('should clear stale results and previous error state before sending new request', () => {
      component.predictionResult = mockSuccessResponse;
      component.errorMessage = 'Previous error';
      component.validationErrors = ['Some validation issue'];

      vi.spyOn(predictionService, 'predictQuality').mockReturnValue(of(mockSuccessResponse));

      component.onSubmit();

      expect(component.errorMessage).toBe('');
      expect(component.validationErrors.length).toBe(0);
      expect(component.predictionResult).toEqual(mockSuccessResponse);
    });

    it('should submit exact 11 features and render predicted_quality and model_type on success', () => {
      const predictSpy = vi
        .spyOn(predictionService, 'predictQuality')
        .mockReturnValue(of(mockSuccessResponse));

      component.onSubmit();
      fixture.detectChanges();

      expect(predictSpy).toHaveBeenCalledTimes(1);
      const passedFeatures = predictSpy.mock.calls[0][0];

      // Verify all 11 features passed to service with exact property names
      expect(passedFeatures.fixed_acidity).toBe(DATASET_BASELINE_VALUES.fixed_acidity);
      expect(passedFeatures.volatile_acidity).toBe(DATASET_BASELINE_VALUES.volatile_acidity);
      expect(passedFeatures.citric_acid).toBe(DATASET_BASELINE_VALUES.citric_acid);
      expect(passedFeatures.residual_sugar).toBe(DATASET_BASELINE_VALUES.residual_sugar);
      expect(passedFeatures.chlorides).toBe(DATASET_BASELINE_VALUES.chlorides);
      expect(passedFeatures.free_sulfur_dioxide).toBe(DATASET_BASELINE_VALUES.free_sulfur_dioxide);
      expect(passedFeatures.total_sulfur_dioxide).toBe(DATASET_BASELINE_VALUES.total_sulfur_dioxide);
      expect(passedFeatures.density).toBe(DATASET_BASELINE_VALUES.density);
      expect(passedFeatures.ph).toBe(DATASET_BASELINE_VALUES.ph);
      expect(passedFeatures.sulphates).toBe(DATASET_BASELINE_VALUES.sulphates);
      expect(passedFeatures.alcohol).toBe(DATASET_BASELINE_VALUES.alcohol);

      expect(component.isLoading).toBe(false);
      expect(component.predictionResult).toEqual(mockSuccessResponse);

      // Verify DOM rendering
      const compiled = fixture.nativeElement as HTMLElement;
      expect(compiled.textContent).toContain('6.24');
      expect(compiled.textContent).toContain('RandomForestRegressor');
      expect(compiled.textContent).toContain('Good Quality');
    });

    it('should handle connection failure (status 0) and display server offline message without fake fallback', () => {
      vi.spyOn(predictionService, 'predictQuality').mockReturnValue(
        throwError(() => ({
          status: 0,
          message: 'Http failure response for http://127.0.0.1:8000/predict: 0 Unknown Error'
        }))
      );

      component.onSubmit();
      fixture.detectChanges();

      expect(component.isLoading).toBe(false);
      expect(component.predictionResult).toBeNull(); // No fake fallback!
      expect(component.errorMessage).toContain('Cannot reach the FastAPI backend');

      const compiled = fixture.nativeElement as HTMLElement;
      expect(compiled.querySelector('app-alert')).toBeTruthy();
      expect(compiled.textContent).toContain('Cannot reach the FastAPI backend');
    });

    it('should handle 422 validation errors with field details from backend', () => {
      vi.spyOn(predictionService, 'predictQuality').mockReturnValue(
        throwError(() => ({
          status: 422,
          error: {
            detail: 'Input validation failed.',
            errors: [
              { field: 'chlorides', message: 'Input should be greater than or equal to 0.005' },
              { field: 'alcohol', message: 'Input should be less than or equal to 20' }
            ]
          }
        }))
      );

      component.onSubmit();
      fixture.detectChanges();

      expect(component.isLoading).toBe(false);
      expect(component.predictionResult).toBeNull();
      expect(component.errorMessage).toContain('rejected the submitted values');
      expect(component.validationErrors.length).toBe(2);
      expect(component.validationErrors[0]).toContain('chlorides');
    });

    it('should handle 500 server error transparently', () => {
      vi.spyOn(predictionService, 'predictQuality').mockReturnValue(
        throwError(() => ({
          status: 500,
          error: { detail: 'Inference evaluation failed due to internal error.' }
        }))
      );

      component.onSubmit();
      fixture.detectChanges();

      expect(component.isLoading).toBe(false);
      expect(component.predictionResult).toBeNull();
      expect(component.errorMessage).toContain('Inference evaluation failed');
    });
  });

  describe('Reset and Baseline Loading', () => {
    it('should reset form to baseline values and clear prediction result and errors', () => {
      // Modify form and set results
      component.predictionForm.patchValue({ alcohol: 14.5, residual_sugar: 8.0 });
      component.predictionResult = mockSuccessResponse;
      component.errorMessage = 'Some error';
      component.validationErrors = ['Error'];

      component.resetForm();

      expect(component.predictionForm.get('alcohol')?.value).toBe(DATASET_BASELINE_VALUES.alcohol);
      expect(component.predictionForm.get('residual_sugar')?.value).toBe(DATASET_BASELINE_VALUES.residual_sugar);
      expect(component.predictionResult).toBeNull();
      expect(component.errorMessage).toBe('');
      expect(component.validationErrors.length).toBe(0);
    });
  });
});
