import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { ApiStatusService } from '../../../core/services/api-status.service';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive],
  templateUrl: './navbar.component.html',
  styleUrls: ['./navbar.component.css'],
})
export class NavbarComponent implements OnInit {
  private api = inject(ApiStatusService);

  apiOnline: boolean | null = null;

  ngOnInit(): void {
    this.api.check().subscribe((ok) => (this.apiOnline = ok));
  }
}
