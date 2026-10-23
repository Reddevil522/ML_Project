import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-empty-state',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './empty-state.component.html',
  styleUrl: './empty-state.component.css'
})
export class EmptyStateComponent {
  @Input() title = 'No records found';
  @Input() description = 'There is currently no data to display for this view.';
  @Input() actionText?: string;
  @Input() icon?: 'wine' | 'analytics' | 'history' = 'history';

  @Output() actionClicked = new EventEmitter<void>();
}
