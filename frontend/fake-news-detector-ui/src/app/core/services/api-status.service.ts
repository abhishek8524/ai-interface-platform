import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { catchError, map, Observable, of } from 'rxjs';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root',
})
export class ApiStatusService {
  private baseUrl = environment.apiBaseUrl;

  constructor(private http: HttpClient) {}

  check(): Observable<boolean> {
    return this.http.get(`${this.baseUrl}/health`).pipe(
      map(() => true),
      catchError(() => of(false))
    );
  }
}
