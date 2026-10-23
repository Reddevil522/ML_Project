import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-badge',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './badge.component.html',
  styleUrl: './badge.component.css'
})
export class BadgeComponent {
  @Input() variant: 'wine' | 'success' | 'warning' | 'danger' | 'neutral' = 'wine';
  @Input() size: 'sm' | 'md' = 'md';
  @Input() showDot = false;
}
