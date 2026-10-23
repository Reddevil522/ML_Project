import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { HistoryService } from '../../core/services/history.service';
import { PredictionHistoryItem, WineFeatures } from '../../core/models/wine.model';
import {
  CardComponent,
  BadgeComponent,
  ButtonComponent,
  AlertComponent,
  EmptyStateComponent
} from '../../shared';

export type SortOrder = 'newest' | 'oldest' | 'quality_desc' | 'quality_asc';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    RouterModule,
    CardComponent,
    BadgeComponent,
    ButtonComponent,
    AlertComponent,
    EmptyStateComponent
  ],
  templateUrl: './history.component.html',
  styleUrl: './history.component.css'
})
export class HistoryComponent implements OnInit {
  allRecords: PredictionHistoryItem[] = [];
  filteredRecords: PredictionHistoryItem[] = [];

  searchQuery = '';
  selectedCategory = 'ALL';
  currentSort: SortOrder = 'newest';

  selectedRecord: PredictionHistoryItem | null = null;
  showClearConfirm = false;

  constructor(
    private historyService: HistoryService,
    private router: Router,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    this.historyService.history$.subscribe((records) => {
      this.allRecords = records;
      this.applyFilters();
      this.cdr.markForCheck();
    });
  }

  applyFilters(): void {
    let result = [...this.allRecords];

    // 1. Text Search Filter (ID, model, category, or quality score)
    const q = this.searchQuery.trim().toLowerCase();
    if (q) {
      result = result.filter((item) => {
        const idMatch = item.id.toLowerCase().includes(q);
        const catMatch = item.prediction.quality_category?.toLowerCase().includes(q);
        const modelMatch = (item.prediction.model_name || item.prediction.model_type)?.toLowerCase().includes(q);
        const scoreMatch = item.prediction.predicted_quality.toString().includes(q);
        const dateMatch = new Date(item.timestamp).toLocaleString().toLowerCase().includes(q);
        return idMatch || catMatch || modelMatch || scoreMatch || dateMatch;
      });
    }

    // 2. Category Filter
    if (this.selectedCategory !== 'ALL') {
      result = result.filter((item) => item.prediction.quality_category === this.selectedCategory);
    }

    // 3. Sorting
    switch (this.currentSort) {
      case 'newest':
        result.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
        break;
      case 'oldest':
        result.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
        break;
      case 'quality_desc':
        result.sort((a, b) => b.prediction.predicted_quality - a.prediction.predicted_quality);
        break;
      case 'quality_asc':
        result.sort((a, b) => a.prediction.predicted_quality - b.prediction.predicted_quality);
        break;
    }

    this.filteredRecords = result;
  }

  clearSearch(): void {
    this.searchQuery = '';
    this.applyFilters();
  }

  resetFilters(): void {
    this.searchQuery = '';
    this.selectedCategory = 'ALL';
    this.currentSort = 'newest';
    this.applyFilters();
  }

  inspectRecord(record: PredictionHistoryItem): void {
    this.selectedRecord = record;
  }

  closeInspection(): void {
    this.selectedRecord = null;
  }

  deleteRecord(id: string): void {
    this.historyService.deleteRecord(id);
    if (this.selectedRecord?.id === id) {
      this.selectedRecord = null;
    }
  }

  deleteSelectedRecord(): void {
    if (this.selectedRecord) {
      this.deleteRecord(this.selectedRecord.id);
    }
  }

  openClearConfirm(): void {
    this.showClearConfirm = true;
  }

  confirmClear(): void {
    this.historyService.clearHistory();
    this.showClearConfirm = false;
    this.selectedRecord = null;
  }

  exportCsv(): void {
    // Export currently filtered records if filtered, otherwise all records
    const exportData = this.filteredRecords.length > 0 ? this.filteredRecords : this.allRecords;
    this.historyService.downloadCsv(exportData);
  }

  navigateToPrediction(): void {
    this.router.navigate(['/prediction']);
  }

  getCategoryVariant(score: number): 'wine' | 'success' | 'warning' | 'danger' | 'neutral' {
    if (score >= 7.0) return 'wine';
    if (score >= 6.0) return 'success';
    if (score >= 5.0) return 'neutral';
    return 'danger';
  }
}
