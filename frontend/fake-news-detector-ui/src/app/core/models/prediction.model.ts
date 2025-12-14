export type PredictionLabel = 'FAKE' | 'REAL' | 'UNCERTAIN';

export interface PredictRequest {
  text: string;
}

export interface PredictResponse {
  prediction: PredictionLabel;
  confidence?: number; // expected 0..100 from backend
  note?: string;       // optional message from backend
}
