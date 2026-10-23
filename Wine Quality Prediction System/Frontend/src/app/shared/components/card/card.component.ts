import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-card',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './card.component.html',
  styleUrl: './card.component.css'
})
export class CardComponent {
  @Input() headerTitle?: string;
  @Input() subtitle?: string;
  @Input() accentBorder = false;
  @Input() elevated = false;
  @Input() hasHeaderSlot = false;
}
