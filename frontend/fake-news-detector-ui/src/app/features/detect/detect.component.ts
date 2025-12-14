import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { finalize } from 'rxjs';

import { PredictionService } from '../../core/services/prediction.service';
import { PredictResponse, PredictionLabel } from '../../core/models/prediction.model';

@Component({
  selector: 'app-detect',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './detect.component.html',
  styleUrls: ['./detect.component.css'],
})
export class DetectComponent {
  private predictionService = inject(PredictionService);

  loading = false;
  errorMsg = '';
  result: PredictResponse | null = null;

  samples = [
    {
      title: 'True example',
      text:
        'The Ministry of Health announced a new vaccination drive starting next month to improve public immunity against seasonal influenza. The program will be carried out across government hospitals and primary health centers.',
    },
    {
      title: 'Fake example',
      text:
        'Scientists confirm that drinking a special herbal tea can instantly cure all types of cancer within 24 hours, according to a secret government report.',
    },
  ];

  form = new FormGroup({
    text: new FormControl<string>('', [
      Validators.required,
      Validators.minLength(3), // ✅ changed to 3
      Validators.maxLength(5000),
    ]),
  });

  get textCtrl() {
    return this.form.controls.text;
  }

  loadSample(i: number) {
    const s = this.samples[i];
    if (!s) return;
    this.form.patchValue({ text: s.text });
    this.result = null;
    this.errorMsg = '';
    this.textCtrl.markAsTouched();
  }

  clear() {
    this.form.reset({ text: '' });
    this.result = null;
    this.errorMsg = '';
    this.loading = false;
  }

  analyze() {
    this.errorMsg = '';
    this.result = null;

    if (this.form.invalid) {
      this.textCtrl.markAsTouched();
      this.errorMsg = 'Please enter valid text before analyzing.';
      return;
    }

    const input = (this.textCtrl.value ?? '').trim();
    this.loading = true;

    this.predictionService
      .predict(input)
      .pipe(finalize(() => (this.loading = false)))
      .subscribe({
        next: (res) => {
          // ✅ normalize prediction
          const rawPred = String(res.prediction ?? '').trim().toUpperCase();

          const pred: PredictionLabel =
            rawPred === 'REAL'
              ? 'REAL'
              : rawPred === 'FAKE'
              ? 'FAKE'
              : rawPred === 'UNCERTAIN'
              ? 'UNCERTAIN'
              : // fallback: handle any unexpected values gracefully
                (rawPred.includes('REAL') || rawPred.includes('TRUE'))
              ? 'REAL'
              : (rawPred.includes('FAKE') || rawPred.includes('FALSE'))
              ? 'FAKE'
              : 'UNCERTAIN';

          // ✅ sanitize confidence (backend sends 0..100)
          const c =
            typeof res.confidence === 'number' && isFinite(res.confidence)
              ? Math.min(Math.max(res.confidence, 0), 100)
              : undefined;

          // ✅ store result exactly in UI shape
          this.result = {
            prediction: pred,
            confidence: c,
            note: res.note,
          };
        },
        error: (err) => {
          const msg =
            err?.error?.detail ||
            err?.message ||
            'API request failed (server down / CORS / network).';
          this.errorMsg = `Could not analyze right now: ${msg}`;
        },
      });
  }

  predictionTitle(p: PredictionLabel): string {
    if (p === 'REAL') return 'Likely Real';
    if (p === 'FAKE') return 'Likely Fake';
    return 'Uncertain';
  }

  confidencePercent(): number {
    return this.result?.confidence != null ? Math.round(this.result.confidence) : 0;
  }
}
