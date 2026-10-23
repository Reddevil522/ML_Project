import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-alert',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './alert.component.html',
  styleUrl: './alert.component.css'
})
export class AlertComponent {
  @Input() type: 'info' | 'success' | 'warning' | 'error' = 'info';
  @Input() title?: string;
  @Input() message?: string;
  @Input() dismissible = false;

  @Output() dismissed = new EventEmitter<void>();

  isDismissed = false;

  dismiss(): void {
    this.isDismissed = true;
    this.dismissed.emit();
  }
}
