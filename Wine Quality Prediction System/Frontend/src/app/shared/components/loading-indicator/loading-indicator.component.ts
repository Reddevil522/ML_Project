import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-loading-indicator',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './loading-indicator.component.html',
  styleUrl: './loading-indicator.component.css'
})
export class LoadingIndicatorComponent {
  @Input() message = 'Loading...';
  @Input() size: 'sm' | 'md' | 'lg' = 'md';
  @Input() fullPage = false;
}
