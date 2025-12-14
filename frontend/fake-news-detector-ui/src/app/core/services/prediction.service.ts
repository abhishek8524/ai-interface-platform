import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';
import { PredictRequest, PredictResponse } from '../models/prediction.model';

@Injectable({
  providedIn: 'root',
})
export class PredictionService {
  private readonly baseUrl = environment.apiBaseUrl;

  constructor(private http: HttpClient) {}

  predict(text: string): Observable<PredictResponse> {
    const url = `${this.baseUrl}/predict`;
    const body: PredictRequest = { text };
    return this.http.post<PredictResponse>(url, body);
  }
}
