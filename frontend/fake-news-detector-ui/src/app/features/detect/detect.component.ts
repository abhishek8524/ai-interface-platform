import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  FormControl,
  FormGroup,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';

type PredictionLabel = 'FAKE' | 'REAL';

interface PredictionResult {
  label: PredictionLabel;
  confidence: number; // 0..1
  note: string;
}

@Component({
  selector: 'app-detect',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './detect.component.html',
  styleUrls: ['./detect.component.css'],
})
export class DetectComponent {
  // --- UI state ---
  loading = false;
  errorMsg = '';
  result: PredictionResult | null = null;

  // --- Demo samples (replace/extend anytime) ---
  samples: { title: string; text: string }[] = [
    {
      title: 'Sample 1: Breaking claim',
      text: `BREAKING: A secret study proves a common household item cures every disease overnight.
Officials are "hiding the truth" and doctors don't want you to know this simple trick.
Share before it gets removed!`,
    },
    {
      title: 'Sample 2: Normal news-style',
      text: `The city council approved a new transit budget on Tuesday after weeks of debate.
The plan includes additional buses during peak hours and funding for station upgrades.`,
    },
  ];

  // --- Reactive Form ---
  form = new FormGroup({
    text: new FormControl<string>('', [
      Validators.required,
      Validators.minLength(40),
      Validators.maxLength(5000),
    ]),
  });

  get textCtrl() {
    return this.form.controls.text;
  }

  // --- Actions ---
  loadSample(i: number) {
    const sample = this.samples[i];
    if (!sample) return;
    this.form.patchValue({ text: sample.text });
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

    // Mock loading (Step 3 will call FastAPI instead)
    this.loading = true;

    const input = (this.textCtrl.value ?? '').trim();

    // Tiny “mock” logic just for UI testing
    const looksFake =
      /breaking|secret|cures|overnight|share before|get removed|don'?t want you to know/i.test(
        input
      );

    const label: PredictionLabel = looksFake ? 'FAKE' : 'REAL';
    const confidence = looksFake ? 0.86 : 0.78;

    setTimeout(() => {
      this.loading = false;
      this.result = {
        label,
        confidence,
        note:
          'This is a UI demo result. In Step 3, this will come from your FastAPI /predict endpoint.',
      };
    }, 700);
  }

  // Helpful for a progress bar %
  confidencePercent(): number {
    return this.result ? Math.round(this.result.confidence * 100) : 0;
  }
}
