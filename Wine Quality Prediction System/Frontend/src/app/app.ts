import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, Router, NavigationEnd } from '@angular/router';
import { filter } from 'rxjs';
import { PredictionService } from './core/services/prediction.service';
import { BackendConnectionStatus } from './core/models/wine.model';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {
  title = 'VinoSense AI';
  isMobileNavOpen = false;
  backendStatus: BackendConnectionStatus = 'checking';

  constructor(
    private predictionService: PredictionService,
    private router: Router,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    this.predictionService.backendStatus$.subscribe((status) => {
      this.backendStatus = status;
      this.cdr.markForCheck();
    });

    // Auto-close mobile drawer upon route navigation
    this.router.events
      .pipe(filter((event) => event instanceof NavigationEnd))
      .subscribe(() => {
        this.isMobileNavOpen = false;
        this.cdr.markForCheck();
      });
  }

  toggleMobileNav(): void {
    this.isMobileNavOpen = !this.isMobileNavOpen;
  }

  closeMobileNav(): void {
    this.isMobileNavOpen = false;
  }

  retryConnection(): void {
    this.predictionService.checkHealth().subscribe({
      next: () => {},
      error: () => {}
    });
  }

  get statusLabel(): string {
    switch (this.backendStatus) {
      case 'connected':
        return 'Backend Connected';
      case 'degraded':
        return 'Model Degraded';
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
