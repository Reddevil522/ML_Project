import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  AbstractControl,
  FormControl,
  FormGroup,
  ReactiveFormsModule,
  ValidationErrors,
  Validators
} from '@angular/forms';
import { finalize } from 'rxjs';
import { PredictionService } from '../../core/services/prediction.service';
import {
  BackendConnectionStatus,
  WineFeatures,
  WinePredictionResponse
} from '../../core/models/wine.model';
import {
  CardComponent,
  ButtonComponent,
  BadgeComponent,
  AlertComponent,
  LoadingIndicatorComponent,
  EmptyStateComponent
} from '../../shared';

/**
 * Cross-field validator ensuring free SO₂ does not exceed total SO₂.
 * Matches backend enological physical consistency rule.
 */
export function sulfurDioxideValidator(group: AbstractControl): ValidationErrors | null {
  const freeControl = group.get('free_sulfur_dioxide');
  const totalControl = group.get('total_sulfur_dioxide');

  if (freeControl && totalControl && freeControl.value != null && totalControl.value != null) {
    const free = Number(freeControl.value);
    const total = Number(totalControl.value);
    if (!isNaN(free) && !isNaN(total) && free > total) {
      return { freeSo2ExceedsTotal: true };
    }
  }
  return null;
}

/**
 * Baseline initial values informed by median empirical distributions of the UCI Red Wine dataset.
 * Provided as an editable baseline — not a verified sample and does not imply a known score.
 */
export const DATASET_BASELINE_VALUES: WineFeatures = {
  fixed_acidity: 7.9,
  volatile_acidity: 0.52,
  citric_acid: 0.26,
  residual_sugar: 2.2,
  chlorides: 0.079,
  free_sulfur_dioxide: 14.0,
  total_sulfur_dioxide: 38.0,
  density: 0.9968,
  ph: 3.31,
  sulphates: 0.62,
  alcohol: 10.2
};

@Component({
  selector: 'app-prediction',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    CardComponent,
    ButtonComponent,
    BadgeComponent,
    AlertComponent,
    LoadingIndicatorComponent,
    EmptyStateComponent
  ],
  templateUrl: './prediction.component.html',
  styleUrl: './prediction.component.css'
})
export class PredictionComponent implements OnInit {
  backendStatus: BackendConnectionStatus = 'checking';
  isLoading = false;
  errorMessage = '';
  validationErrors: string[] = [];
  predictionResult: WinePredictionResponse | null = null;
  lastLatencyMs: number | null = null;

  /**
   * Reactive form with exact 11 physicochemical features and defensible enological bounds.
   */
  readonly predictionForm: FormGroup = new FormGroup(
    {
      fixed_acidity: new FormControl<number | null>(DATASET_BASELINE_VALUES.fixed_acidity, [
        Validators.required,
        Validators.min(3.0),
        Validators.max(25.0)
      ]),
      volatile_acidity: new FormControl<number | null>(DATASET_BASELINE_VALUES.volatile_acidity, [
        Validators.required,
        Validators.min(0.05),
        Validators.max(3.0)
      ]),
      citric_acid: new FormControl<number | null>(DATASET_BASELINE_VALUES.citric_acid, [
        Validators.required,
        Validators.min(0.0),
        Validators.max(2.0)
      ]),
      residual_sugar: new FormControl<number | null>(DATASET_BASELINE_VALUES.residual_sugar, [
        Validators.required,
        Validators.min(0.5),
        Validators.max(100.0)
      ]),
      chlorides: new FormControl<number | null>(DATASET_BASELINE_VALUES.chlorides, [
        Validators.required,
        Validators.min(0.005),
        Validators.max(1.0)
      ]),
      free_sulfur_dioxide: new FormControl<number | null>(DATASET_BASELINE_VALUES.free_sulfur_dioxide, [
        Validators.required,
        Validators.min(0),
        Validators.max(150)
      ]),
      total_sulfur_dioxide: new FormControl<number | null>(DATASET_BASELINE_VALUES.total_sulfur_dioxide, [
        Validators.required,
        Validators.min(0),
        Validators.max(450)
      ]),
      density: new FormControl<number | null>(DATASET_BASELINE_VALUES.density, [
        Validators.required,
        Validators.min(0.9800),
        Validators.max(1.0400)
      ]),
      ph: new FormControl<number | null>(DATASET_BASELINE_VALUES.ph ?? 3.31, [
        Validators.required,
        Validators.min(2.5),
        Validators.max(4.5)
      ]),
      sulphates: new FormControl<number | null>(DATASET_BASELINE_VALUES.sulphates, [
        Validators.required,
        Validators.min(0.1),
        Validators.max(3.0)
      ]),
      alcohol: new FormControl<number | null>(DATASET_BASELINE_VALUES.alcohol, [
        Validators.required,
        Validators.min(6.0),
        Validators.max(20.0)
      ])
    },
    { validators: [sulfurDioxideValidator] }
  );

  constructor(
    public predictionService: PredictionService,
    private cdr: ChangeDetectorRef
  ) { }

  ngOnInit(): void {
    this.predictionService.backendStatus$.subscribe((status) => {
      this.backendStatus = status;
      this.cdr.markForCheck();
    });
  }

  get offlineNoticeMessage(): string {
    return `The ML prediction service at ${this.predictionService.apiBaseUrl} is currently offline. You can still edit chemical attributes, but submitting inference requires starting the backend server.`;
  }

  get hasSulfurDioxideError(): boolean {
    return (
      this.predictionForm.hasError('freeSo2ExceedsTotal') &&
      (this.predictionForm.get('free_sulfur_dioxide')?.touched ||
        this.predictionForm.get('total_sulfur_dioxide')?.touched ||
        false)
    );
  }

  isFieldInvalid(fieldName: string): boolean {
    const control = this.predictionForm.get(fieldName);
    return !!(control && control.invalid && (control.dirty || control.touched));
  }

  getFieldErrorMessage(
    fieldName: string,
    label: string,
    min: number,
    max: number,
    unit: string
  ): string {
    const control = this.predictionForm.get(fieldName);
    if (!control || !control.errors) return '';

    if (control.errors['required']) {
      return `${label} is required.`;
    }
    if (control.errors['min']) {
      return `${label} must be at least ${min} ${unit}`.trim();
    }
    if (control.errors['max']) {
      return `${label} cannot exceed ${max} ${unit}`.trim();
    }
    return `Invalid value for ${label}.`;
  }

  onSubmit(): void {
    // Prevent duplicate submissions and invalid submissions
    if (this.isLoading) {
      return;
    }

    if (this.predictionForm.invalid) {
      this.predictionForm.markAllAsTouched();
      return;
    }

    // Clear stale results and previous error state upon starting new prediction
    this.isLoading = true;
    this.errorMessage = '';
    this.validationErrors = [];
    this.predictionResult = null;
    this.lastLatencyMs = null;
    this.cdr.markForCheck();

    const val = this.predictionForm.getRawValue();
    const features: WineFeatures = {
      fixed_acidity: Number(val.fixed_acidity),
      volatile_acidity: Number(val.volatile_acidity),
      citric_acid: Number(val.citric_acid),
      residual_sugar: Number(val.residual_sugar),
      chlorides: Number(val.chlorides),
      free_sulfur_dioxide: Number(val.free_sulfur_dioxide),
      total_sulfur_dioxide: Number(val.total_sulfur_dioxide),
      density: Number(val.density),
      ph: Number(val.ph),
      sulphates: Number(val.sulphates),
      alcohol: Number(val.alcohol)
    };

    const startTime = performance.now();

    this.predictionService.predictQuality(features).pipe(
      finalize(() => {
        this.isLoading = false;
        this.cdr.markForCheck();
      })
    ).subscribe({
      next: (res) => {
        this.predictionResult = res;
        this.lastLatencyMs = Math.round(performance.now() - startTime);
        this.cdr.markForCheck();
      },
      error: (err) => {
        // Never fabricate a fallback score; display the actual error transparently
        this.predictionResult = null;

        if (err.status === 0 || err.status === 504) {
          this.errorMessage = `Cannot reach the FastAPI backend at ${this.predictionService.apiBaseUrl}. Please ensure the server is running.`;
        } else if (err.status === 422) {
          const detail = err.error?.detail;
          const errors = err.error?.errors;
          if (Array.isArray(errors) && errors.length > 0) {
            this.errorMessage = 'The model rejected the submitted values due to validation constraints:';
            this.validationErrors = errors.map((e: any) => `${e.field}: ${e.message}`);
          } else if (typeof detail === 'string') {
            this.errorMessage = detail;
          } else {
            this.errorMessage = 'Validation error: please verify all chemical inputs meet required ranges.';
          }
        } else if (err.status === 503) {
          this.errorMessage =
            err.error?.detail ||
            'Model artifact is not loaded on the server. Please run "python src/train.py".';
        } else {
          this.errorMessage =
            err.error?.detail ||
            err.message ||
            'An unexpected error occurred during prediction evaluation.';
        }
        this.cdr.markForCheck();
      }
    });
  }

  loadBaselineValues(): void {
    this.predictionForm.reset(DATASET_BASELINE_VALUES);
    this.predictionResult = null;
    this.errorMessage = '';
    this.validationErrors = [];
    this.lastLatencyMs = null;
  }

  resetForm(): void {
    this.loadBaselineValues();
  }

  clearErrors(): void {
    this.errorMessage = '';
    this.validationErrors = [];
  }
}
