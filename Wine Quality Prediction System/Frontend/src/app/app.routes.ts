import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'dashboard',
    pathMatch: 'full'
  },
  {
    path: 'dashboard',
    loadComponent: () =>
      import('./features/dashboard/dashboard.component').then((m) => m.DashboardComponent),
    title: 'Dashboard | VinoSense AI'
  },
  {
    path: 'prediction',
    loadComponent: () =>
      import('./features/prediction/prediction.component').then((m) => m.PredictionComponent),
    title: 'Sensory Prediction | VinoSense AI'
  },
  {
    path: 'analytics',
    loadComponent: () =>
      import('./features/analytics/analytics.component').then((m) => m.AnalyticsComponent),
    title: 'Model Analytics | VinoSense AI'
  },
  {
    path: 'history',
    loadComponent: () =>
      import('./features/history/history.component').then((m) => m.HistoryComponent),
    title: 'Inference History | VinoSense AI'
  },
  {
    path: '**',
    redirectTo: 'dashboard'
  }
];
